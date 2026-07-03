#!/usr/bin/env python3
"""Fit and evaluate simple runtime value calibration transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_by_ply_metrics,
    compute_per_opening_metrics,
    compute_seat_metrics,
    parse_game_jsonl,
)
from ml.alphazero_lite.run_post_schedule_pairwise_puct_update import fmt  # noqa: E402
from ml.alphazero_lite.run_post_schedule_value_trust_calibration import (  # noqa: E402
    budget_pair_label,
    ensure_suite_prefixes,
    heldout_lane_summary,
    load_suite_entries,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
    runtime_cost_summary,
    spearman_correlation,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    build_input_summary,
    read_jsonl,
    require_existing_file,
    sha256_file,
    verify_expected_hash,
)
from ml.alphazero_lite.runtime_root_sensitivity import (  # noqa: E402
    root_sensitivity_prefilter as shared_root_sensitivity_prefilter,
    runtime_sensitivity_diagnostic_for_opening_suite,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_eval_search_options,
    build_search_profile,
)
from ml.alphazero_lite.value_transforms import (  # noqa: E402
    affine_tanh_transform,
    clamp_unit_value,
    effective_value_transform_config,
    normalize_value_transform_config,
    value_transform_hash,
)

SUMMARY_SCHEMA = "azlite_post_schedule_value_calibration_transform_v1"
SUMMARY_FILENAME = "summary_metrics.json"
FIT_REPORT_FILENAME = "value_calibration_fit_report.json"
MANIFEST_FILENAME = "value_transform_manifest.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-post-schedule-value-calibration-transform-results.md"
)
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
BOOTSTRAP_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")


@dataclass(frozen=True)
class LaneConfig:
    name: str
    description: str
    value_transform: dict[str, Any] | None


def canonical_requested_transform(token: str) -> str:
    normalized = str(token).strip()
    return "identity_ref" if normalized == "identity" else normalized


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--calibration-states", required=True)
    parser.add_argument("--calibration-audit", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument(
        "--transforms",
        default=(
            "identity,global_affine_tanh,opening_only_affine_tanh,phase_affine_tanh,"
            "conservative_opening_affine"
        ),
    )
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--gate-games", type=int, default=120)
    parser.add_argument(
        "--root-sensitivity-budgets",
        default="384:256,768:768",
    )
    parser.add_argument(
        "--root-sensitivity-min-move-change-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--root-sensitivity-min-mean-abs-value-delta",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    return parser.parse_args()


def sign_value(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def parse_budget_labels(text: str) -> list[str]:
    return [
        budget_pair_label(challenger, current)
        for challenger, current in parse_budget_pairs(text)
    ]


def manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def value_transform_args(value_transform: dict[str, Any] | None) -> list[str]:
    if value_transform is None:
        return []
    return [
        "--value-transform-json",
        json.dumps(normalize_value_transform_config(value_transform), sort_keys=True),
    ]


def build_lane_search_profile(
    *,
    lane: LaneConfig,
    challenger_sims: int,
    current_sims: int,
    effective_c_puct: float,
) -> dict[str, Any]:
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        value_transform=lane.value_transform,
    )
    return build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(int(challenger_sims), int(current_sims)),
        c_puct=float(effective_c_puct),
        search_options=search_options,
        extra_fields={
            "challenger_simulations": int(challenger_sims),
            "current_simulations": int(current_sims),
        },
    )


def suite_group_key(row: dict[str, Any]) -> str:
    prefix_moves = row.get("prefix_moves", [])
    return (
        f"{row.get('suite', 'unknown')}::{'/'.join(str(int(m)) for m in prefix_moves)}"
    )


def split_calibration_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted(
        {suite_group_key(row) for row in rows},
        key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest(),
    )
    train_group_count = max(1, int(round(len(groups) * 0.7)))
    train_groups = set(groups[:train_group_count])
    train_rows = [row for row in rows if suite_group_key(row) in train_groups]
    validation_rows = [row for row in rows if suite_group_key(row) not in train_groups]
    if not validation_rows:
        validation_rows = train_rows[-max(1, len(train_rows) // 3) :]
        train_rows = train_rows[: len(train_rows) - len(validation_rows)]
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_group_count": len(train_groups),
        "validation_group_count": len(groups) - len(train_groups),
        "group_count": len(groups),
        "group_hash": manifest_hash({"groups": groups}),
    }


def rows_for_phase(rows: list[dict[str, Any]], phase_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("phase")) == phase_name]


def summarize_predictions(
    rows: list[dict[str, Any]], predictions: list[float]
) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mse": None,
            "mae": None,
            "sign_accuracy": None,
            "spearman": None,
        }
    outcomes = [float(row["continuation_outcome"]) for row in rows]
    mse = statistics.fmean(
        (pred - outcome) ** 2
        for pred, outcome in zip(predictions, outcomes, strict=True)
    )
    mae = statistics.fmean(
        abs(pred - outcome) for pred, outcome in zip(predictions, outcomes, strict=True)
    )
    sign_accuracy = statistics.fmean(
        1.0 if sign_value(pred) == sign_value(outcome) else 0.0
        for pred, outcome in zip(predictions, outcomes, strict=True)
    )
    return {
        "count": len(rows),
        "mse": float(mse),
        "mae": float(mae),
        "sign_accuracy": float(sign_accuracy),
        "spearman": spearman_correlation(predictions, outcomes),
    }


def summarize_transform_metrics(
    *, rows: list[dict[str, Any]], transform: dict[str, Any] | None
) -> dict[str, Any]:
    predictions = [
        apply_row_transform(float(row["raw_value"]), str(row["phase"]), transform)
        for row in rows
    ]
    by_phase = {}
    for phase_name in ("opening", "mid", "late"):
        phase_rows = rows_for_phase(rows, phase_name)
        phase_predictions = [
            apply_row_transform(float(row["raw_value"]), str(row["phase"]), transform)
            for row in phase_rows
        ]
        by_phase[phase_name] = summarize_predictions(phase_rows, phase_predictions)
    return {
        "overall": summarize_predictions(rows, predictions),
        "by_phase": by_phase,
    }


def apply_row_transform(
    value: float, phase_name: str, transform: dict[str, Any] | None
) -> float:
    normalized = normalize_value_transform_config(transform)
    if normalized is None or normalized["kind"] == "identity":
        return float(value)
    phase_key = "midgame" if phase_name == "mid" else phase_name
    params = normalized["phase_params"].get(phase_key)
    if params is None:
        return float(value)
    if normalized["kind"] == "affine_tanh":
        return affine_tanh_transform(value, a=float(params["a"]), b=float(params["b"]))
    x_values = [float(item) for item in params["x"]]
    y_values = [float(item) for item in params["y"]]
    clamped = clamp_unit_value(value)
    if len(x_values) == 1:
        return float(y_values[0])
    if clamped <= x_values[0]:
        return float(y_values[0])
    if clamped >= x_values[-1]:
        return float(y_values[-1])
    for index in range(1, len(x_values)):
        if clamped > x_values[index]:
            continue
        left_x = x_values[index - 1]
        right_x = x_values[index]
        left_y = y_values[index - 1]
        right_y = y_values[index]
        if right_x <= left_x:
            return float(right_y)
        mix = (clamped - left_x) / (right_x - left_x)
        return float(left_y + (mix * (right_y - left_y)))
    return float(y_values[-1])


def affine_objective(
    *, a: float, b: float, values: np.ndarray, outcomes: np.ndarray, reg_lambda: float
) -> float:
    transformed = np.tanh((a * np.arctanh(np.clip(values, -0.999999, 0.999999))) + b)
    mse = float(np.mean(np.square(transformed - outcomes)))
    regularizer = reg_lambda * (((a - 1.0) ** 2) + (b**2))
    return mse + regularizer


def fit_affine_params(
    rows: list[dict[str, Any]], *, allow_negative_a: bool = False
) -> dict[str, float] | None:
    if not rows:
        return None
    values = np.asarray([float(row["raw_value"]) for row in rows], dtype=np.float64)
    outcomes = np.asarray(
        [float(row["continuation_outcome"]) for row in rows], dtype=np.float64
    )
    a_low = -0.5 if allow_negative_a else 0.25
    a_high = 2.5
    b_low = -1.0
    b_high = 1.0
    best = {
        "a": 1.0,
        "b": 0.0,
        "objective": affine_objective(
            a=1.0, b=0.0, values=values, outcomes=outcomes, reg_lambda=0.02
        ),
    }
    for _ in range(4):
        a_grid = np.linspace(a_low, a_high, 41)
        b_grid = np.linspace(b_low, b_high, 41)
        for a_value in a_grid:
            z_values = np.tanh(
                (a_value * np.arctanh(np.clip(values, -0.999999, 0.999999)))
                + b_grid[:, None]
            )
            errors = np.mean(np.square(z_values - outcomes), axis=1) + (
                0.02 * (((a_value - 1.0) ** 2) + np.square(b_grid))
            )
            best_index = int(np.argmin(errors))
            objective = float(errors[best_index])
            if objective < float(best["objective"]):
                best = {
                    "a": float(a_value),
                    "b": float(b_grid[best_index]),
                    "objective": objective,
                }
        a_center = float(best["a"])
        b_center = float(best["b"])
        a_span = max(0.15, (a_high - a_low) / 6.0)
        b_span = max(0.10, (b_high - b_low) / 6.0)
        a_low = max(-0.5 if allow_negative_a else 0.05, a_center - a_span)
        a_high = min(3.0, a_center + a_span)
        b_low = max(-1.5, b_center - b_span)
        b_high = min(1.5, b_center + b_span)
    if abs(float(best["a"])) > 3.0 or abs(float(best["b"])) > 1.5:
        return None
    return {"a": float(best["a"]), "b": float(best["b"])}


def fit_isotonic_phase(rows: list[dict[str, Any]]) -> dict[str, list[float]] | None:
    if not rows:
        return None
    ordered = sorted(
        (
            (
                clamp_unit_value(float(row["raw_value"])),
                float(row["continuation_outcome"]),
            )
            for row in rows
        ),
        key=lambda item: item[0],
    )
    blocks: list[dict[str, Any]] = []
    for value, outcome in ordered:
        if blocks and math.isclose(value, float(blocks[-1]["max_x"]), abs_tol=1e-12):
            block = blocks[-1]
            block["sum_y"] += outcome
            block["count"] += 1
            block["mean_y"] = block["sum_y"] / block["count"]
            continue
        blocks.append(
            {
                "min_x": value,
                "max_x": value,
                "sum_y": outcome,
                "count": 1,
                "mean_y": outcome,
            }
        )
    index = 0
    while index < len(blocks) - 1:
        if float(blocks[index]["mean_y"]) <= float(blocks[index + 1]["mean_y"]):
            index += 1
            continue
        merged = {
            "min_x": float(blocks[index]["min_x"]),
            "max_x": float(blocks[index + 1]["max_x"]),
            "sum_y": float(blocks[index]["sum_y"]) + float(blocks[index + 1]["sum_y"]),
            "count": int(blocks[index]["count"]) + int(blocks[index + 1]["count"]),
        }
        merged["mean_y"] = merged["sum_y"] / merged["count"]
        blocks[index : index + 2] = [merged]
        if index > 0:
            index -= 1
    x_values: list[float] = []
    y_values: list[float] = []
    for block in blocks:
        x_values.append(float(block["min_x"]))
        y_values.append(float(max(-1.0, min(1.0, block["mean_y"]))))
        if float(block["max_x"]) > float(block["min_x"]):
            x_values.append(float(block["max_x"]))
            y_values.append(float(max(-1.0, min(1.0, block["mean_y"]))))
    return {"x": x_values, "y": y_values}


def build_affine_transform(
    *,
    name: str,
    opening: dict[str, float],
    midgame: dict[str, float],
    late: dict[str, float],
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "affine_tanh",
        "phase_params": {
            "opening": {"a": float(opening["a"]), "b": float(opening["b"])},
            "midgame": {"a": float(midgame["a"]), "b": float(midgame["b"])},
            "late": {"a": float(late["a"]), "b": float(late["b"])},
        },
    }


def fit_transforms(
    *, rows: list[dict[str, Any]], requested_tokens: list[str], workdir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, LaneConfig]]:
    split = split_calibration_rows(rows)
    train_rows = split["train_rows"]
    validation_rows = split["validation_rows"]
    identity_metrics = summarize_transform_metrics(rows=validation_rows, transform=None)
    transforms: dict[str, dict[str, Any]] = {
        "identity_ref": {
            "name": "identity_ref",
            "supported": True,
            "diagnostic_only": False,
            "value_transform": None,
            "validation": identity_metrics,
            "improves_validation": True,
        }
    }

    global_params = fit_affine_params(train_rows)
    if global_params is not None:
        global_transform = build_affine_transform(
            name="global_affine_tanh",
            opening=global_params,
            midgame=global_params,
            late=global_params,
        )
        transforms["global_affine_tanh"] = {
            "name": "global_affine_tanh",
            "supported": True,
            "diagnostic_only": False,
            "value_transform": global_transform,
            "validation": summarize_transform_metrics(
                rows=validation_rows, transform=global_transform
            ),
        }

    opening_params = fit_affine_params(rows_for_phase(train_rows, "opening"))
    if opening_params is not None:
        identity_params = {"a": 1.0, "b": 0.0}
        opening_transform = build_affine_transform(
            name="opening_only_affine_tanh",
            opening=opening_params,
            midgame=identity_params,
            late=identity_params,
        )
        transforms["opening_only_affine_tanh"] = {
            "name": "opening_only_affine_tanh",
            "supported": True,
            "diagnostic_only": False,
            "value_transform": opening_transform,
            "validation": summarize_transform_metrics(
                rows=validation_rows, transform=opening_transform
            ),
        }
        conservative_params = {
            "a": 1.0 + (0.5 * (float(opening_params["a"]) - 1.0)),
            "b": 0.5 * float(opening_params["b"]),
        }
        conservative_transform = build_affine_transform(
            name="conservative_opening_affine",
            opening=conservative_params,
            midgame=identity_params,
            late=identity_params,
        )
        transforms["conservative_opening_affine"] = {
            "name": "conservative_opening_affine",
            "supported": True,
            "diagnostic_only": False,
            "value_transform": conservative_transform,
            "validation": summarize_transform_metrics(
                rows=validation_rows, transform=conservative_transform
            ),
        }

    phase_params: dict[str, dict[str, float]] = {}
    for phase_name in ("opening", "mid", "late"):
        params = fit_affine_params(rows_for_phase(train_rows, phase_name))
        if params is None:
            phase_params = {}
            break
        phase_params["midgame" if phase_name == "mid" else phase_name] = params
    if phase_params:
        phase_transform = build_affine_transform(
            name="phase_affine_tanh",
            opening=phase_params["opening"],
            midgame=phase_params["midgame"],
            late=phase_params["late"],
        )
        transforms["phase_affine_tanh"] = {
            "name": "phase_affine_tanh",
            "supported": True,
            "diagnostic_only": False,
            "value_transform": phase_transform,
            "validation": summarize_transform_metrics(
                rows=validation_rows, transform=phase_transform
            ),
        }

    isotonic_phase_params: dict[str, Any] = {}
    for phase_name in ("opening", "mid", "late"):
        params = fit_isotonic_phase(rows_for_phase(train_rows, phase_name))
        if params is None:
            isotonic_phase_params = {}
            break
        isotonic_phase_params["midgame" if phase_name == "mid" else phase_name] = params
    transforms["phase_isotonic"] = {
        "name": "phase_isotonic",
        "supported": bool(isotonic_phase_params),
        "diagnostic_only": False,
        "value_transform": None
        if not isotonic_phase_params
        else {
            "name": "phase_isotonic",
            "kind": "phase_isotonic",
            "phase_params": isotonic_phase_params,
        },
        "validation": None,
    }
    if isotonic_phase_params:
        transforms["phase_isotonic"]["validation"] = summarize_transform_metrics(
            rows=validation_rows,
            transform=transforms["phase_isotonic"]["value_transform"],
        )

    sign_probe_params = fit_affine_params(
        rows_for_phase(train_rows, "opening"), allow_negative_a=True
    )
    transforms["opening_sign_bias_probe"] = {
        "name": "opening_sign_bias_probe",
        "supported": sign_probe_params is not None,
        "diagnostic_only": True,
        "value_transform": None,
        "validation": None,
    }
    if sign_probe_params is not None:
        sign_probe_transform = build_affine_transform(
            name="opening_sign_bias_probe",
            opening=sign_probe_params,
            midgame={"a": 1.0, "b": 0.0},
            late={"a": 1.0, "b": 0.0},
        )
        transforms["opening_sign_bias_probe"]["value_transform"] = sign_probe_transform
        transforms["opening_sign_bias_probe"]["validation"] = (
            summarize_transform_metrics(
                rows=validation_rows,
                transform=sign_probe_transform,
            )
        )

    identity_mse = float(identity_metrics["overall"]["mse"] or 0.0)
    identity_mae = float(identity_metrics["overall"]["mae"] or 0.0)
    manifest: dict[str, Any] = {
        "schema": "azlite_value_transform_manifest_v1",
        "date": str(date.today()),
        "split": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_group_count": split["train_group_count"],
            "validation_group_count": split["validation_group_count"],
            "group_count": split["group_count"],
            "group_hash": split["group_hash"],
        },
        "transforms": {},
    }
    lanes: dict[str, LaneConfig] = {
        "identity_ref": LaneConfig(
            name="identity_ref",
            description="Current promoted default without a runtime value transform.",
            value_transform=None,
        )
    }
    for token, entry in transforms.items():
        validation = entry.get("validation")
        improves_validation = False
        if validation is not None:
            mse = float(validation["overall"].get("mse") or identity_mse)
            mae = float(validation["overall"].get("mae") or identity_mae)
            improves_validation = (mse < identity_mse - 1e-6) or (
                math.isclose(mse, identity_mse, abs_tol=1e-6)
                and mae < identity_mae - 1e-6
            )
        entry["improves_validation"] = improves_validation or token == "identity_ref"
        normalized_transform = normalize_value_transform_config(
            entry.get("value_transform")
        )
        manifest["transforms"][token] = {
            "supported": bool(entry.get("supported", False)),
            "diagnostic_only": bool(entry.get("diagnostic_only", False)),
            "improves_validation": bool(entry["improves_validation"]),
            "value_transform": normalized_transform,
            "value_transform_hash": value_transform_hash(
                normalized_transform, identity_name=token
            ),
            "validation": validation,
        }
        if (
            token in requested_tokens
            and token != "identity_ref"
            and bool(entry.get("supported", False))
            and bool(entry["improves_validation"])
            and not bool(entry.get("diagnostic_only", False))
        ):
            lanes[token] = LaneConfig(
                name=token,
                description=token.replace("_", " "),
                value_transform=normalized_transform,
            )
    fit_report = {
        "schema": "azlite_value_transform_fit_report_v1",
        "date": str(date.today()),
        "split": manifest["split"],
        "requested_transforms": requested_tokens,
        "identity_validation": identity_metrics,
        "transforms": manifest["transforms"],
    }
    (workdir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (workdir / FIT_REPORT_FILENAME).write_text(
        json.dumps(fit_report, indent=2), encoding="utf-8"
    )
    return manifest, fit_report, lanes


def run_opening_suite_lane(
    *,
    lane: LaneConfig,
    suite_name: str,
    suite_path: Path,
    suite_entries: list[dict[str, Any]],
    current_path: Path,
    pair: tuple[int, int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
    workdir: Path,
) -> dict[str, Any]:
    challenger_sims, current_sims = pair
    label = budget_pair_label(challenger_sims, current_sims)
    pair_dir = workdir / suite_name / lane.name / label.replace(":", "_")
    pair_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = pair_dir / "metrics.json"
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=default_c_puct,
    )
    transform_hash = value_transform_hash(lane.value_transform, identity_name=lane.name)
    lane_search_profile = build_lane_search_profile(
        lane=lane,
        challenger_sims=challenger_sims,
        current_sims=current_sims,
        effective_c_puct=effective_c_puct,
    )
    cache_context = {
        "lane": lane.name,
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "current_path": str(current_path),
        "current_sha256": sha256_file(current_path / "weights.json"),
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "effective_c_puct": effective_c_puct,
        "c_puct_schedule": schedule_definition(
            default_c_puct=default_c_puct, schedule=cpuct_schedule
        )["overrides"],
        "games_per_opening": games_per_opening,
        "workers": workers,
        "seed": seed,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": tactical_root_bias,
        "value_transform": lane.value_transform,
        "value_transform_hash": transform_hash,
        "search_profile_hash": lane_search_profile["hash"],
    }
    if metrics_path.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_context") == cache_context
            and cached.get("search_profile_hash") == lane_search_profile["hash"]
        ):
            return cached

    total_games = len(suite_entries) * max(1, int(games_per_opening))
    all_game_entries: list[dict[str, Any]] = []
    seat_reports: list[dict[str, Any]] = []
    for seat in (0, 1):
        seat_dir = pair_dir / f"starts_{seat}"
        seat_dir.mkdir(parents=True, exist_ok=True)
        arena_json = seat_dir / "arena.json"
        games_jsonl = seat_dir / "games.jsonl"
        suite_jsonl = seat_dir / "opening_suite.jsonl"
        meta_path = seat_dir / "metadata.json"
        ensure_suite_prefixes(suite_jsonl, suite_entries)
        seat_context = {**cache_context, "challenger_starts": seat}
        if meta_path.is_file() and arena_json.is_file() and games_jsonl.is_file():
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("cache_context") == seat_context:
                seat_reports.append(json.loads(arena_json.read_text(encoding="utf-8")))
                all_game_entries.extend(parse_game_jsonl(str(games_jsonl)))
                continue

        cmd = [
            _python(),
            str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
            "--challenger",
            str(current_path),
            "--current",
            str(current_path),
            "--challenger-simulations",
            str(challenger_sims),
            "--current-simulations",
            str(current_sims),
            "--games",
            str(total_games),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            "--min-score",
            "0.0",
            "--out",
            str(arena_json),
            "--game-jsonl",
            str(games_jsonl),
            "--challenger-starts",
            str(seat),
            "--games-per-opening",
            str(games_per_opening),
            "--opening-prefixes-jsonl",
            str(suite_jsonl),
            "--root-policy-mode",
            "deterministic",
            "--c-puct",
            str(effective_c_puct),
            "--tactical-root-bias",
            str(tactical_root_bias),
            *value_transform_args(lane.value_transform),
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
            raise RuntimeError(
                f"arena failed for {lane.name} {suite_name} {label}: {result.stderr[-2000:]}"
            )
        report = json.loads(arena_json.read_text(encoding="utf-8"))
        game_entries = parse_game_jsonl(str(games_jsonl))
        seat_reports.append(report)
        all_game_entries.extend(game_entries)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_context": seat_context,
                    "games": len(game_entries),
                    "arena_score": report.get("score"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics = compute_seat_metrics(all_game_entries)
    per_opening_metrics = compute_per_opening_metrics(all_game_entries)
    by_ply_metrics = compute_by_ply_metrics(all_game_entries)
    mean_move_values = [
        float(report.get("notes", {}).get("move_time_mean_ms"))
        for report in seat_reports
        if report.get("notes", {}).get("move_time_mean_ms") is not None
    ]
    p95_move_values = [
        float(report.get("notes", {}).get("move_time_p95_ms"))
        for report in seat_reports
        if report.get("notes", {}).get("move_time_p95_ms") is not None
    ]
    payload = {
        **metrics,
        "cache_context": cache_context,
        "suite_name": suite_name,
        "budget_pair": label,
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "effective_c_puct": effective_c_puct,
        "tactical_root_bias": tactical_root_bias,
        "transform_name": lane.name,
        "transform_parameters": effective_value_transform_config(
            lane.value_transform, identity_name=lane.name
        ),
        "value_transform_hash": transform_hash,
        "per_opening_metrics": per_opening_metrics,
        "by_ply_metrics": {str(key): value for key, value in by_ply_metrics.items()},
        "search_profile": lane_search_profile,
        "search_profile_hash": lane_search_profile["hash"],
        "move_time_mean_ms": statistics.fmean(mean_move_values)
        if mean_move_values
        else None,
        "move_time_p95_ms": statistics.fmean(p95_move_values)
        if p95_move_values
        else None,
        "total_games": len(all_game_entries),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_suite_benchmark(
    *,
    lanes: list[LaneConfig],
    suite_path: Path,
    current_path: Path,
    budget_pairs: list[tuple[int, int]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    games_per_opening: int,
    workers: int,
    seed: int,
    timeout: int,
    workdir: Path,
) -> dict[str, Any]:
    suite_entries = load_suite_entries(suite_path)
    report: dict[str, Any] = {
        "suite_path": str(suite_path),
        "suite_name": suite_path.stem,
        "suite_sha256": sha256_file(suite_path),
        "suite_size": len(suite_entries),
        "lanes": {},
    }
    for lane in lanes:
        lane_result: dict[str, Any] = {"candidate": lane.name, "budget_results": {}}
        for pair in budget_pairs:
            budget_result = run_opening_suite_lane(
                lane=lane,
                suite_name=suite_path.stem,
                suite_path=suite_path,
                suite_entries=suite_entries,
                current_path=current_path,
                pair=pair,
                default_c_puct=default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=tactical_root_bias,
                games_per_opening=games_per_opening,
                workers=workers,
                seed=seed,
                timeout=timeout,
                workdir=workdir,
            )
            lane_result["budget_results"][budget_result["budget_pair"]] = budget_result
        report["lanes"][lane.name] = lane_result
    return report


def suite_delta(
    report: dict[str, Any], lane_name: str, default_name: str, budget: str
) -> float:
    lane_ds = float(report["lanes"][lane_name]["budget_results"][budget]["ds"])
    default_ds = float(report["lanes"][default_name]["budget_results"][budget]["ds"])
    return lane_ds - default_ds


def balanced_score(report: dict[str, Any], lane_name: str, default_name: str) -> float:
    return (
        suite_delta(report, lane_name, default_name, "384:256")
        + (0.5 * suite_delta(report, lane_name, default_name, "768:768"))
        + (0.25 * suite_delta(report, lane_name, default_name, "1200:1200"))
        + (0.25 * suite_delta(report, lane_name, default_name, "1200:256"))
    )


def carry_lanes_from_medium(report: dict[str, Any], default_name: str) -> list[str]:
    selected: set[str] = {default_name}
    ranked = sorted(
        (name for name in report["lanes"] if name != default_name),
        key=lambda name: (balanced_score(report, name, default_name), name),
        reverse=True,
    )
    for lane_name in ranked:
        delta_384 = suite_delta(report, lane_name, default_name, "384:256")
        delta_768 = suite_delta(report, lane_name, default_name, "768:768")
        delta_1200 = suite_delta(report, lane_name, default_name, "1200:1200")
        delta_1200_256 = suite_delta(report, lane_name, default_name, "1200:256")
        if delta_384 >= 0.05 and delta_768 >= -0.05:
            selected.add(lane_name)
        if (delta_1200 >= 0.05 or delta_1200_256 >= 0.05) and delta_384 >= -0.05:
            selected.add(lane_name)
    for lane_name in ranked:
        non_reference = [name for name in selected if name != default_name]
        if len(non_reference) >= 5:
            break
        selected.add(lane_name)
    non_reference = sorted(
        [name for name in selected if name != default_name],
        key=lambda name: (balanced_score(report, name, default_name), name),
        reverse=True,
    )[:5]
    return [default_name, *non_reference]


def carry_lanes_from_fixed_large(
    report: dict[str, Any], default_name: str
) -> list[str]:
    ranked = sorted(
        (name for name in report["lanes"] if name != default_name),
        key=lambda name: (balanced_score(report, name, default_name), name),
        reverse=True,
    )
    selected: set[str] = {default_name, *ranked[:3]}
    for lane_name in ranked:
        delta_384 = suite_delta(report, lane_name, default_name, "384:256")
        delta_768 = suite_delta(report, lane_name, default_name, "768:768")
        delta_1200 = suite_delta(report, lane_name, default_name, "1200:1200")
        delta_1200_256 = suite_delta(report, lane_name, default_name, "1200:256")
        robustness_regression = (
            delta_768 < -0.05 or delta_1200 < -0.03 or delta_1200_256 < -0.03
        )
        if delta_384 >= 0.05 and not robustness_regression:
            selected.add(lane_name)
    return [default_name, *sorted(name for name in selected if name != default_name)]


def build_bootstrap_rows(
    *,
    heldout_reports: dict[str, dict[str, Any]],
    lane_names: list[str],
    default_name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    suite_rows = {
        suite_name: {
            "candidates": {
                lane_name: {"budget_results": lane_report["budget_results"]}
                for lane_name, lane_report in report["lanes"].items()
            }
        }
        for suite_name, report in heldout_reports.items()
    }
    result: dict[str, dict[str, Any]] = {}
    for lane_name in lane_names:
        if lane_name == default_name:
            continue
        result[lane_name] = {}
        for budget_pair in BOOTSTRAP_BUDGETS:
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a=lane_name,
                candidate_b=default_name,
                budget_pair=budget_pair,
                metric_key="ds",
            )
            result[lane_name][budget_pair] = bootstrap_ci(
                diffs,
                seed=seed + abs(hash((lane_name, budget_pair))) % 100000,
                samples=bootstrap_samples,
            )
    return result


def aggregate_p0_p1(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lane_name: {
            "p0_score": lane_report["budget_results"]["384:256"].get("p0_score"),
            "p1_score": lane_report["budget_results"]["384:256"].get("p1_score"),
            "gap": lane_report["budget_results"]["384:256"].get("ds"),
        }
        for lane_name, lane_report in report["lanes"].items()
    }


def aggregate_duplicates(report: dict[str, Any]) -> dict[str, Any]:
    return {
        lane_name: lane_report["budget_results"]["384:256"].get(
            "duplicate_trajectory_count"
        )
        for lane_name, lane_report in report["lanes"].items()
    }


def run_gate(
    *,
    lane: LaneConfig,
    current_path: Path,
    out_path: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
    games: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        str(current_path),
        "--current-path",
        str(current_path),
        "--out",
        str(out_path),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        DEFAULT_BUDGET_PAIRS,
        "--c-puct",
        str(default_c_puct),
        "--c-puct-schedule-json",
        json.dumps(cpuct_schedule, sort_keys=True),
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        str(tactical_root_bias),
        *value_transform_args(lane.value_transform),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed for {lane.name}: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def runtime_sensitivity_diagnostic(
    *,
    current_path: Path,
    suite_path: Path,
    lanes: list[LaneConfig],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
    workdir: Path,
    budget_labels: list[str],
) -> dict[str, Any]:
    return runtime_sensitivity_diagnostic_for_opening_suite(
        current_path=current_path,
        suite_path=suite_path,
        lane_specs=[
            {"name": lane.name, "value_transform": lane.value_transform}
            for lane in lanes
        ],
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        seed=seed,
        workdir=workdir,
        budget_labels=budget_labels,
    )


def root_sensitivity_prefilter(
    *,
    runtime_sensitivity: dict[str, Any],
    lane_names: list[str],
    default_name: str,
    min_move_change_rate: float,
    min_mean_abs_value_delta: float,
) -> dict[str, Any]:
    return shared_root_sensitivity_prefilter(
        runtime_sensitivity=runtime_sensitivity,
        lane_names=lane_names,
        default_name=default_name,
        min_move_change_rate=min_move_change_rate,
        min_mean_abs_value_delta=min_mean_abs_value_delta,
    )


def classify_lane(
    *,
    lane_name: str,
    heldout_summary: dict[str, Any],
    bootstrap: dict[str, Any],
    gate_results: dict[str, Any],
) -> str:
    if lane_name == "identity_ref":
        return "reference"
    delta_384 = float(heldout_summary["deltas_vs_default"].get("384:256") or 0.0)
    delta_768 = float(heldout_summary["deltas_vs_default"].get("768:768") or 0.0)
    delta_1200 = float(heldout_summary["deltas_vs_default"].get("1200:1200") or 0.0)
    delta_1200_256 = float(heldout_summary["deltas_vs_default"].get("1200:256") or 0.0)
    delta_768_256 = float(heldout_summary["deltas_vs_default"].get("768:256") or 0.0)
    delta_256_768 = float(heldout_summary["deltas_vs_default"].get("256:768") or 0.0)
    ci_384 = bootstrap.get("384:256", {})
    ci_1200 = bootstrap.get("1200:1200", {})
    gate_ok = True
    if lane_name in gate_results:
        gate_ok = (
            gate_results[lane_name].get("classification") == "high_search_breakthrough"
        )
    improved_384 = delta_384 >= 0.05 and float(ci_384.get("lower") or 0.0) > 0.01
    improved_1200 = delta_1200 >= 0.08 and float(ci_1200.get("lower") or 0.0) > 0.01
    major_regression = (
        delta_768 < -0.05
        or delta_768_256 < -0.03
        or delta_1200_256 < -0.03
        or delta_256_768 < -0.03
        or delta_1200 < -0.03
    )
    if (improved_384 or improved_1200) and not major_regression and gate_ok:
        return "value_calibration_runtime_candidate"
    if delta_384 > 0.0 and (delta_768 < -0.05 or delta_1200 < -0.03):
        return "value_transform_tradeoff"
    if delta_384 <= 0.0 and delta_1200 <= 0.0:
        return "value_calibration_overfit"
    return "value_calibration_not_enough"


def write_report(summary: dict[str, Any], lanes: list[LaneConfig]) -> None:
    default_name = lanes[0].name
    fit_report = summary["fit_report"]
    medium = summary["medium"]
    fixed_large = summary["fixed_large"]
    heldout_means = summary["heldout_mean_summary"]
    bootstrap = summary["bootstrap"]
    runtime_cost = summary["runtime_cost"]
    gate_results = summary["gate_results"]
    runtime_sensitivity = summary["runtime_sensitivity"]
    root_sensitivity_prefilter_summary = summary["root_sensitivity_prefilter"]

    validation_rows = []
    for lane_name, entry in fit_report["transforms"].items():
        validation = entry.get("validation")
        if validation is None:
            continue
        overall = validation["overall"]
        validation_rows.append(
            [
                lane_name,
                "overall",
                overall["count"],
                fmt(overall["mse"]),
                fmt(overall["mae"]),
                fmt(overall["sign_accuracy"]),
                fmt(overall["spearman"]),
            ]
        )
        for phase_name in ("opening", "mid", "late"):
            phase_metrics = validation["by_phase"][phase_name]
            validation_rows.append(
                [
                    lane_name,
                    phase_name,
                    phase_metrics["count"],
                    fmt(phase_metrics["mse"]),
                    fmt(phase_metrics["mae"]),
                    fmt(phase_metrics["sign_accuracy"]),
                    fmt(phase_metrics["spearman"]),
                ]
            )

    transform_rows = []
    for lane_name, entry in fit_report["transforms"].items():
        transform_rows.append(
            [
                lane_name,
                entry["supported"],
                entry["improves_validation"],
                entry["value_transform_hash"],
                json.dumps(entry["value_transform"], sort_keys=True),
            ]
        )

    lane_flow_rows = [
        ["requested_runtime_tokens", ", ".join(summary["requested_transform_tokens"])],
        [
            "validation_supported_runtime_lanes",
            ", ".join(summary["validated_runtime_lanes"]),
        ],
        [
            "root_sensitivity_retained_lanes",
            ", ".join(summary["root_sensitivity_retained_lanes"]),
        ],
        [
            "root_sensitivity_filtered_lanes",
            ", ".join(summary["root_sensitivity_filtered_lanes"]),
        ],
        ["medium_carried_lanes", ", ".join(summary["medium_carried_lanes"])],
        ["fixed_large_carried_lanes", ", ".join(summary["fixed_large_carried_lanes"])],
        ["heldout_carried_lanes", ", ".join(summary["heldout_carried_lanes"])],
    ]
    root_prefilter_rows = []
    for lane_name, row in root_sensitivity_prefilter_summary["decisions"].items():
        root_prefilter_rows.append(
            [
                lane_name,
                row.get("retain"),
                row.get("reason"),
                fmt(row.get("max_move_change_rate")),
                fmt(row.get("max_mean_abs_value_delta")),
            ]
        )

    def budget_table(report: dict[str, Any]) -> str:
        rows = []
        for lane in lanes:
            if lane.name not in report["lanes"]:
                continue
            budgets = report["lanes"][lane.name]["budget_results"]
            rows.append(
                [
                    lane.name,
                    fmt(budgets["384:256"]["ds"]),
                    fmt(budgets["768:256"]["ds"]),
                    fmt(budgets["768:768"]["ds"]),
                    fmt(budgets["1200:1200"]["ds"]),
                    fmt(budgets["1200:256"]["ds"]),
                    fmt(budgets["256:768"]["ds"]),
                ]
            )
        return markdown_table(
            [
                "Lane",
                "384:256",
                "768:256",
                "768:768",
                "1200:1200",
                "1200:256",
                "256:768",
            ],
            rows,
        )

    heldout_rows = []
    for lane_name, lane_summary in heldout_means.items():
        heldout_rows.append(
            [
                lane_name,
                fmt(lane_summary["384:256"]["mean_ds"]),
                fmt(lane_summary["384:256"]["worst_suite_ds"]),
                fmt(lane_summary["1200:1200"]["mean_ds"]),
                fmt(lane_summary["deltas_vs_default"]["384:256"]),
                fmt(lane_summary["deltas_vs_default"]["1200:1200"]),
                summary["lane_classifications"].get(lane_name),
            ]
        )

    bootstrap_rows = []
    for lane_name, budget_map in bootstrap.items():
        for budget_pair in BOOTSTRAP_BUDGETS:
            ci = budget_map[budget_pair]
            bootstrap_rows.append(
                [
                    f"{lane_name} minus {default_name} @ {budget_pair}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )

    p0_rows = [
        [lane_name, fmt(row["p0_score"]), fmt(row["p1_score"]), fmt(row["gap"])]
        for lane_name, row in summary["p0_p1_split_384_256"].items()
    ]
    dup_rows = [
        [lane_name, fmt(value)]
        for lane_name, value in summary["duplicate_trajectory_count"].items()
    ]
    runtime_rows = [
        [
            lane_name,
            fmt(row["mean_move_latency_ms"]),
            fmt(row["p95_move_latency_ms"]),
            fmt(row["relative_slowdown"], digits=3),
        ]
        for lane_name, row in runtime_cost.items()
    ]
    gate_rows = [
        [lane_name, report.get("classification")]
        for lane_name, report in gate_results.items()
    ]
    sensitivity_rows = []
    for budget_pair, lane_map in runtime_sensitivity["budgets"].items():
        for lane_name, row in lane_map.items():
            if lane_name == "identity_ref":
                continue
            sensitivity_rows.append(
                [
                    budget_pair,
                    lane_name,
                    row["move_change_count"],
                    fmt(row["move_change_rate"]),
                    fmt(row["mean_abs_value_delta"]),
                    fmt(row["max_abs_value_delta"]),
                ]
            )

    report_text = "\n".join(
        [
            "# AlphaZero-Lite Post-Schedule Value Calibration Transform Results",
            "",
            f"**Date**: {summary['date']}",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Artifact Hash",
            "",
            f"- Current artifact: `{summary['inputs']['current_artifact']['path']}`",
            f"- Current weights SHA256: `{summary['inputs']['current_artifact']['sha256']}`",
            f"- Expected SHA256: `{summary['inputs']['current_hash_verification']['expected_sha256']}`",
            "",
            "## Promoted Search Schedule Confirmation",
            "",
            f"- Runtime profile: `{json.dumps(summary['promoted_runtime_profile'], sort_keys=True)}`",
            "",
            "## Calibration-State Split And Hashes",
            "",
            f"- Calibration states: `{summary['inputs']['calibration_states']['path']}` (`{summary['inputs']['calibration_states']['sha256']}`)",
            f"- Calibration audit: `{summary['inputs']['calibration_audit']['path']}` (`{summary['inputs']['calibration_audit']['sha256']}`)",
            f"- Split: `{json.dumps(summary['fit_report']['split'], sort_keys=True)}`",
            "",
            "## Transform Definitions And Parameters",
            "",
            markdown_table(
                ["Transform", "Supported", "Improves val", "Hash", "Parameters"],
                transform_rows,
            ),
            "",
            "## Lane Flow",
            "",
            markdown_table(["Stage", "Lanes"], lane_flow_rows),
            "",
            "## Validation Calibration Table",
            "",
            markdown_table(
                ["Transform", "Slice", "Count", "MSE", "MAE", "Sign acc", "Spearman"],
                validation_rows,
            ),
            "",
            "## Root Sensitivity Prefilter",
            "",
            f"- Thresholds: `{json.dumps(root_sensitivity_prefilter_summary['thresholds'], sort_keys=True)}`",
            "",
            markdown_table(
                [
                    "Lane",
                    "Retain",
                    "Reason",
                    "Max move change rate",
                    "Max mean abs value delta",
                ],
                root_prefilter_rows,
            ),
            "",
            "## Medium DS Table",
            "",
            budget_table(medium),
            "",
            "## Fixed Large DS Table",
            "",
            budget_table(fixed_large),
            "",
            "## Held-Out Mean/Worst-Suite DS Table",
            "",
            markdown_table(
                [
                    "Lane",
                    "Held-out mean 384:256",
                    "Worst-suite 384:256",
                    "Held-out mean 1200:1200",
                    "Delta 384:256",
                    "Delta 1200:1200",
                    "Classification",
                ],
                heldout_rows,
            ),
            "",
            "## Bootstrap CI",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%"], bootstrap_rows
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(["Lane", "Mean P0", "Mean P1", "Gap"], p0_rows),
            "",
            "## Duplicate Trajectory Count",
            "",
            markdown_table(["Lane", "Mean duplicates"], dup_rows),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                ["Lane", "Mean move latency", "P95 move latency", "Relative slowdown"],
                runtime_rows,
            ),
            "",
            "## Gate Classification",
            "",
            markdown_table(["Lane", "Classification"], gate_rows),
            "",
            "## Runtime Sensitivity Diagnostic",
            "",
            markdown_table(
                [
                    "Budget",
                    "Lane",
                    "Move changes",
                    "Move change rate",
                    "Mean abs value delta",
                    "Max abs value delta",
                ],
                sensitivity_rows,
            ),
            "",
        ]
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = Path(args.current)
    require_existing_file(current_path / "weights.json", "current weights")
    require_existing_file(Path(args.calibration_states), "calibration states")
    require_existing_file(Path(args.calibration_audit), "calibration audit")
    require_existing_file(Path(args.medium_suite), "medium suite")
    require_existing_file(Path(args.fixed_large_suite), "fixed large suite")
    for path in parse_csv_paths(args.heldout_suites):
        require_existing_file(path, f"heldout suite {path.name}")

    current_hash_info = verify_expected_hash(
        current_path / "weights.json",
        args.expected_current_weights_sha256,
        "current artifact",
    )
    calibration_rows = list(read_jsonl(Path(args.calibration_states)))
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    requested_tokens = [
        token.strip() for token in args.transforms.split(",") if token.strip()
    ]
    requested_lane_names = [
        canonical_requested_transform(token) for token in requested_tokens
    ]
    manifest, fit_report, lane_map = fit_transforms(
        rows=calibration_rows,
        requested_tokens=requested_lane_names,
        workdir=workdir,
    )
    runtime_lane_names = [
        "identity_ref",
        *[
            name
            for name in requested_lane_names
            if name in lane_map and name != "identity_ref"
        ],
    ]
    lanes = [lane_map[name] for name in runtime_lane_names]
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    root_sensitivity_budget_labels = parse_budget_labels(args.root_sensitivity_budgets)

    runtime_sensitivity = runtime_sensitivity_diagnostic(
        current_path=current_path,
        suite_path=Path(args.medium_suite),
        lanes=lanes,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        seed=int(args.seed),
        workdir=workdir,
        budget_labels=root_sensitivity_budget_labels,
    )
    root_sensitivity_prefilter_summary = root_sensitivity_prefilter(
        runtime_sensitivity=runtime_sensitivity,
        lane_names=runtime_lane_names,
        default_name="identity_ref",
        min_move_change_rate=float(args.root_sensitivity_min_move_change_rate),
        min_mean_abs_value_delta=float(args.root_sensitivity_min_mean_abs_value_delta),
    )
    sensitivity_lane_names = root_sensitivity_prefilter_summary["retain"]
    sensitivity_lanes = [
        lane_map[name] for name in sensitivity_lane_names if name in lane_map
    ]

    medium_report = run_suite_benchmark(
        lanes=sensitivity_lanes,
        suite_path=Path(args.medium_suite),
        current_path=current_path,
        budget_pairs=budget_pairs,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        games_per_opening=int(args.games_per_opening),
        workers=int(args.workers),
        seed=int(args.seed),
        timeout=int(args.timeout),
        workdir=workdir / "medium",
    )
    carried_medium_names = [
        name
        for name in carry_lanes_from_medium(medium_report, "identity_ref")
        if name in root_sensitivity_prefilter_summary["retain"]
    ]
    fixed_large_lanes = [
        lane_map[name] for name in carried_medium_names if name in lane_map
    ]
    fixed_large_report = run_suite_benchmark(
        lanes=fixed_large_lanes,
        suite_path=Path(args.fixed_large_suite),
        current_path=current_path,
        budget_pairs=budget_pairs,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        games_per_opening=int(args.games_per_opening),
        workers=int(args.workers),
        seed=int(args.seed),
        timeout=int(args.timeout),
        workdir=workdir / "fixed_large",
    )
    carried_heldout_names = carry_lanes_from_fixed_large(
        fixed_large_report, "identity_ref"
    )
    heldout_lanes = [
        lane_map[name] for name in carried_heldout_names if name in lane_map
    ]
    heldout_reports = {
        suite_path.stem: run_suite_benchmark(
            lanes=heldout_lanes,
            suite_path=suite_path,
            current_path=current_path,
            budget_pairs=budget_pairs,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            games_per_opening=int(args.games_per_opening),
            workers=int(args.workers),
            seed=int(args.seed),
            timeout=int(args.timeout),
            workdir=workdir / "heldout" / suite_path.stem,
        )
        for suite_path in parse_csv_paths(args.heldout_suites)
    }
    heldout_means = {
        lane.name: heldout_lane_summary(
            heldout_reports=heldout_reports,
            lane_name=lane.name,
            default_name="identity_ref",
            budget_pairs=list(
                fixed_large_report["lanes"]["identity_ref"]["budget_results"].keys()
            ),
        )
        for lane in heldout_lanes
    }
    bootstrap = build_bootstrap_rows(
        heldout_reports=heldout_reports,
        lane_names=[lane.name for lane in heldout_lanes],
        default_name="identity_ref",
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    gate_candidates = {"identity_ref"}
    for lane_name, lane_summary in heldout_means.items():
        if lane_name == "identity_ref":
            continue
        delta_384 = float(lane_summary["deltas_vs_default"].get("384:256") or 0.0)
        delta_768 = float(lane_summary["deltas_vs_default"].get("768:768") or 0.0)
        delta_1200 = float(lane_summary["deltas_vs_default"].get("1200:1200") or 0.0)
        delta_1200_256 = float(lane_summary["deltas_vs_default"].get("1200:256") or 0.0)
        if (
            (delta_384 >= 0.05 or delta_1200 >= 0.08)
            and delta_768 >= -0.05
            and delta_1200_256 >= -0.03
        ):
            gate_candidates.add(lane_name)
    gate_results = {
        lane_name: run_gate(
            lane=lane_map[lane_name],
            current_path=current_path,
            out_path=workdir / "gate" / f"{lane_name}.json",
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=int(args.seed),
            workers=int(args.workers),
            games=int(args.gate_games),
        )
        for lane_name in sorted(gate_candidates)
    }

    lane_classifications = {
        lane_name: classify_lane(
            lane_name=lane_name,
            heldout_summary=lane_summary,
            bootstrap=bootstrap.get(lane_name, {}),
            gate_results=gate_results,
        )
        for lane_name, lane_summary in heldout_means.items()
    }
    non_reference_classes = [
        label
        for lane_name, label in lane_classifications.items()
        if lane_name != "identity_ref"
    ]
    if "value_calibration_runtime_candidate" in non_reference_classes:
        classification = "value_calibration_runtime_candidate"
    elif "value_transform_tradeoff" in non_reference_classes:
        classification = "value_transform_tradeoff"
    elif "value_calibration_overfit" in non_reference_classes:
        classification = "value_calibration_overfit"
    else:
        classification = "value_calibration_not_enough"

    summary = {
        "schema": SUMMARY_SCHEMA,
        "date": str(date.today()),
        "classification": classification,
        "lane_classifications": lane_classifications,
        "inputs": {
            "current_artifact": build_input_summary(current_path / "weights.json"),
            "current_hash_verification": current_hash_info,
            "calibration_states": build_input_summary(Path(args.calibration_states)),
            "calibration_audit": build_input_summary(Path(args.calibration_audit)),
            "medium_suite": build_input_summary(Path(args.medium_suite)),
            "fixed_large_suite": build_input_summary(Path(args.fixed_large_suite)),
            "heldout_suites": [
                build_input_summary(path)
                for path in parse_csv_paths(args.heldout_suites)
            ],
        },
        "promoted_runtime_profile": {
            "artifact": str(current_path),
            "default_c_puct": float(args.default_c_puct),
            "c_puct_schedule": schedule_definition(
                default_c_puct=float(args.default_c_puct),
                schedule=cpuct_schedule,
            ),
            "tactical_root_bias": float(args.tactical_root_bias),
            "root_policy_mode": "deterministic",
            "root_prior_transform": None,
            "search_mode": "full",
        },
        "requested_transform_tokens": requested_tokens,
        "validated_runtime_lanes": runtime_lane_names,
        "root_sensitivity_retained_lanes": root_sensitivity_prefilter_summary["retain"],
        "root_sensitivity_filtered_lanes": root_sensitivity_prefilter_summary[
            "filtered_out"
        ],
        "medium_carried_lanes": carried_medium_names,
        "fixed_large_carried_lanes": [lane.name for lane in fixed_large_lanes],
        "heldout_carried_lanes": [lane.name for lane in heldout_lanes],
        "manifest": manifest,
        "fit_report": fit_report,
        "medium": medium_report,
        "fixed_large": fixed_large_report,
        "heldout": heldout_reports,
        "heldout_mean_summary": heldout_means,
        "bootstrap": bootstrap,
        "p0_p1_split_384_256": aggregate_p0_p1(fixed_large_report),
        "duplicate_trajectory_count": aggregate_duplicates(fixed_large_report),
        "runtime_cost": runtime_cost_summary(fixed_large_report, "identity_ref"),
        "gate_results": gate_results,
        "runtime_sensitivity": runtime_sensitivity,
        "root_sensitivity_prefilter": root_sensitivity_prefilter_summary,
        "workdir": str(workdir),
    }
    (workdir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(summary, heldout_lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
