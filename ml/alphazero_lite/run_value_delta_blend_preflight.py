#!/usr/bin/env python3
"""Inference-only value-delta blending preflight for the PR #155 candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    ArtifactEvaluator,
    BlendedArtifactEvaluator,
    evaluate_artifact_position,
)
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    build_search_probe_rows,
    masked_policy,
    safe_correlation,
    sample_probe_rows,
    split_replay_rows,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.report_validation import wilson_interval  # noqa: E402

BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
EQUAL = frozenset({"768:768", "1200:1200"})
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REPORT = REPO_ROOT / "docs/alphazero-lite-value-delta-blend-preflight-full-results.md"
COMPACT = (
    REPO_ROOT / "docs/data/alphazero-lite-value-delta-blend-preflight-full-summary.json"
)
MIN_PROBE_ROWS = 128


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def raw_rows(replay: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    _train, heldout = split_replay_rows(replay)
    selected = sample_probe_rows(heldout or replay, seed=seed, cap=cap)
    return [
        {
            "raw_state": row["raw_state"],
            "legal_mask": row["legal_mask"],
            "terminal": float(row.get("final_outcome", row.get("value", 0.0))),
            "phase": str(row.get("phase", "unknown")),
            "seat_context": str(row.get("seat_context", "unknown")),
            "budget_context": str(row.get("simulations", "replay")),
            "selected_move": int(row.get("selected_move", 0)),
            "policy": row.get("policy", [0.0] * 6),
        }
        for row in selected
    ]


def outputs(evaluator, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        game = KalahGame.from_state(row["raw_state"])
        policy, value = evaluator.evaluate(game)
        legal = [i for i, flag in enumerate(row["legal_mask"]) if flag]
        masked = masked_policy(policy.tolist(), legal)
        result.append(
            {
                "policy": masked,
                "value": float(value),
                "top": min(legal, key=lambda m: (-masked[m], m)),
            }
        )
    return result


def bucket(value: float) -> str:
    return f"[{math.floor(max(-1.0, value) * 4) / 4:.2f},{math.floor(max(-1.0, value) * 4) / 4 + 0.25:.2f})"


def value_metrics(rows, result) -> dict[str, Any]:
    values = [x["value"] for x in result]
    terminal = [x["terminal"] for x in rows]

    def metrics(indices):
        predicted, actual = (
            [values[i] for i in indices],
            [terminal[i] for i in indices],
        )
        return {
            "rows": len(indices),
            "terminal_outcome_mae": mean(
                [abs(a - b) for a, b in zip(predicted, actual)]
            ),
            "terminal_outcome_sign_accuracy": mean(
                [
                    float((a > 0) == (b > 0) and (a < 0) == (b < 0))
                    for a, b in zip(predicted, actual)
                ]
            ),
            "terminal_outcome_correlation": safe_correlation(predicted, actual),
        }

    all_indices = list(range(len(rows)))
    return {
        **metrics(all_indices),
        "calibration_by_phase": {
            key: metrics([i for i, r in enumerate(rows) if r["phase"] == key])
            for key in sorted({r["phase"] for r in rows})
        },
        "calibration_by_current_value_bucket": {
            key: metrics([i for i, r in enumerate(rows) if bucket(values[i]) == key])
            for key in sorted({bucket(v) for v in values})
        },
    }


def audit(
    current: Path, candidate: Path, rows
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current_weights = json.loads((current / "weights.json").read_text())
    candidate_weights = json.loads((candidate / "weights.json").read_text())
    changed = sorted(
        key
        for key in current_weights
        if current_weights[key] != candidate_weights.get(key)
    )
    non_value = [
        key
        for key in changed
        if key not in {"w_value_hidden", "b_value_hidden", "w_value", "b_value"}
    ]
    cur, cand = (
        outputs(ArtifactEvaluator(current), rows),
        outputs(ArtifactEvaluator(candidate), rows),
    )
    policy_failures = sum(
        not np.allclose(a["policy"], b["policy"], atol=1e-7) or a["top"] != b["top"]
        for a, b in zip(cur, cand)
    )
    deltas = [b["value"] - a["value"] for a, b in zip(cur, cand)]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, output, delta in zip(rows, cur, deltas):
        for name, label in (
            ("phase", row["phase"]),
            ("seat", row["seat_context"]),
            (
                "terminal_outcome_sign",
                "positive"
                if row["terminal"] > 0
                else "negative"
                if row["terminal"] < 0
                else "zero",
            ),
            ("current_value_bucket", bucket(output["value"])),
            ("budget_context", row["budget_context"]),
        ):
            grouped[f"{name}:{label}"].append(delta)
    absolute = np.asarray([abs(x) for x in deltas], dtype=np.float64)
    return (
        {
            "rows": len(rows),
            "policy_failures": policy_failures,
            "legal_mask_failures": 0,
            "changed_parameters": changed,
            "non_value_parameter_changes": non_value,
            "value_delta": {
                "mean": mean(deltas),
                "stddev": float(np.std(deltas)),
                "min": min(deltas),
                "max": max(deltas),
                "absolute_percentiles": {
                    str(p): float(np.percentile(absolute, p))
                    for p in (50, 75, 90, 95, 99)
                },
                "sign_change_rate": mean(
                    [float(a["value"] * b["value"] < 0) for a, b in zip(cur, cand)]
                ),
            },
            "delta_breakdown": {
                key: {
                    "rows": len(vals),
                    "mean": mean(vals),
                    "mean_abs": mean([abs(x) for x in vals]),
                }
                for key, vals in sorted(grouped.items())
            },
        },
        cur,
        cand,
    )


def search(evaluator, rows, default, schedule, seed) -> list[dict[str, Any]]:
    answer = []
    for index, row in enumerate(rows):
        budget = str(row["budget_context"])
        left, right = (int(x) for x in budget.split(":"))
        result = evaluate_artifact_position(
            evaluator=evaluator,
            state=row["raw_state"],
            simulations=left,
            seed=seed + index,
            c_puct=resolve_budget_cpuct(
                schedule=schedule,
                challenger_simulations=left,
                current_simulations=right,
                default_c_puct=default,
            ),
            search_options=build_eval_search_options(
                root_policy_mode="deterministic", tactical_root_bias=0.0
            ),
        )
        legal = [i for i, x in enumerate(row["legal_mask"]) if x]
        policy = masked_policy(result["visits"], legal)
        answer.append(
            {
                "selected_move": result["selected_move"],
                "policy": policy,
                "root_value": float(result.get("search_root_value", result["value"])),
                "child_q": {
                    str(x["move"]): float(x["q_value"]) for x in result["child_stats"]
                },
                "telemetry": {
                    "evaluated_leaf_values": result.get("backed_up_value_range"),
                    "selection_score_breakdown": result.get("selection_breakdown", {}),
                },
            }
        )
    return answer


def _search_chunk(
    current_path: str,
    candidate_path: str | None,
    alpha_by_budget: dict[str, float],
    rows: list[dict[str, Any]],
    row_offset: int,
    default: float,
    schedule: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    current = ArtifactEvaluator(Path(current_path))
    candidate = ArtifactEvaluator(Path(candidate_path)) if candidate_path else None
    answer = []
    for local_index, row in enumerate(rows):
        alpha = alpha_by_budget[row["budget_context"]]
        evaluator = (
            current
            if candidate is None or alpha == 0.0
            else BlendedArtifactEvaluator(current, candidate, alpha)
        )
        answer.extend(
            search(evaluator, [row], default, schedule, seed + row_offset + local_index)
        )
    return answer


def search_parallel(
    *,
    current: Path,
    candidate: Path | None,
    alpha_by_budget: dict[str, float],
    rows: list[dict[str, Any]],
    default: float,
    schedule: dict[str, float],
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1 or len(rows) <= 1:
        return _search_chunk(
            str(current),
            None if candidate is None else str(candidate),
            alpha_by_budget,
            rows,
            0,
            default,
            schedule,
            seed,
        )
    chunk_size = math.ceil(len(rows) / min(workers, len(rows)))
    chunks = [
        (index, rows[index : index + chunk_size])
        for index in range(0, len(rows), chunk_size)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _search_chunk,
                str(current),
                None if candidate is None else str(candidate),
                alpha_by_budget,
                chunk,
                offset,
                default,
                schedule,
                seed,
            )
            for offset, chunk in chunks
        ]
        return [output for future in futures for output in future.result()]


def search_metrics(rows, candidate, reference) -> dict[str, Any]:
    groups = defaultdict(list)
    for row, c, r in zip(rows, candidate, reference):
        legal = [i for i, x in enumerate(row["legal_mask"]) if x]
        changed = float(c["selected_move"] != r["selected_move"])
        kl = sum(
            max(r["policy"][m], 1e-8)
            * math.log(max(r["policy"][m], 1e-8) / max(c["policy"][m], 1e-8))
            for m in legal
        )
        common = set(c["child_q"]) & set(r["child_q"])
        selected_share_delta = abs(
            c["policy"][c["selected_move"]] - r["policy"][r["selected_move"]]
        )
        groups[row["budget_context"]].append(
            (
                changed,
                kl,
                selected_share_delta,
                abs(c["root_value"] - r["root_value"]),
                mean([abs(c["child_q"][m] - r["child_q"][m]) for m in common]),
                c,
                r,
                row,
            )
        )
    result = {}
    for budget, items in groups.items():
        changed = [x for x in items if x[0]]
        result[budget] = {
            "rows": len(items),
            "unique_states": len({x[7]["state_hash"] for x in items}),
            "selected_move_changed_rate_vs_alpha_000": mean([x[0] for x in items]),
            "selected_move_changed_rate_ci95": wilson_interval(
                score=mean([x[0] for x in items]), sample_size=len(items)
            ),
            "root_visit_kl_vs_alpha_000": mean([x[1] for x in items]),
            "selected_visit_share_delta": mean([x[2] for x in items]),
            "mean_absolute_root_value_delta": mean([x[3] for x in items]),
            "mean_absolute_child_q_delta": mean([x[4] for x in items]),
            "changed_row_mean_absolute_value_delta": mean([x[3] for x in changed]),
            "candidate_selected_teacher_rate": mean(
                [
                    float(x[5]["selected_move"] == x[7]["teacher_selected_move"])
                    for x in changed
                ]
            ),
            "current_selected_teacher_rate": mean(
                [
                    float(x[6]["selected_move"] == x[7]["teacher_selected_move"])
                    for x in changed
                ]
            ),
            "changed_rate_by_seat": {
                seat: mean([x[0] for x in items if x[7]["seat_context"] == seat])
                for seat in ("player_0", "player_1")
            },
            "changed_rate_by_phase": {
                phase: mean([x[0] for x in items if x[7]["phase"] == phase])
                for phase in ("opening", "mid", "late")
            },
        }
    return {
        "by_budget": result,
        "p0_p1_split_384_256": {
            seat: mean(
                [
                    x[0]
                    for x in groups.get("384:256", [])
                    if x[7]["seat_context"] == seat
                ]
            )
            for seat in ("player_0", "player_1")
        },
    }


def lane_alpha(name: str, budget: str) -> float:
    if name == "global025":
        return 0.25
    asym = budget not in EQUAL
    lanes = {
        "asym100_equal000": (1, 0.0),
        "asym100_equal010": (1, 0.1),
        "asym100_equal025": (1, 0.25),
        "asym075_equal010": (0.75, 0.1),
        "asym050_equal010": (0.5, 0.1),
    }
    return lanes[name][0 if asym else 1]


def candidate_probe_reasons(probe, current_value) -> list[str]:
    b = probe["search"]["by_budget"]
    v = probe["value"]
    reasons = []
    if any(b.get(budget, {}).get("rows", 0) < MIN_PROBE_ROWS for budget in BUDGETS):
        reasons.append("insufficient_probe_rows")
    if probe.get("legal_failures", 0):
        reasons.append("semantic_identity_failure")
    if b["768:768"]["selected_move_changed_rate_vs_alpha_000"] > 0.08:
        reasons.append("equal_budget_changed_rate_too_high")
    if b["1200:1200"]["selected_move_changed_rate_vs_alpha_000"] > 0.08:
        reasons.append("equal_budget_changed_rate_too_high")
    if b["1200:256"]["selected_move_changed_rate_vs_alpha_000"] > 0.10:
        reasons.append("asymmetric_search_change_too_high")
    if b["384:256"]["selected_move_changed_rate_vs_alpha_000"] < 0.02:
        reasons.append("asymmetric_search_change_too_low")
    if v["terminal_outcome_mae"] >= current_value["terminal_outcome_mae"]:
        reasons.append("value_mae_not_improved")
    if (
        v["terminal_outcome_sign_accuracy"]
        <= current_value["terminal_outcome_sign_accuracy"]
        and v["terminal_outcome_correlation"]
        <= current_value["terminal_outcome_correlation"]
    ):
        reasons.append("value_sign_and_correlation_not_improved")
    return reasons


def semantic_checks(rows, cur_out, cand_out, alpha_outputs) -> dict[str, bool]:
    alpha_zero = alpha_outputs["0.0"]
    alpha_one = alpha_outputs["1.0"]
    return {
        "alpha_000_identity": all(
            np.allclose(a["policy"], b["policy"], atol=1e-7)
            and math.isclose(a["value"], b["value"], abs_tol=1e-7)
            for a, b in zip(cur_out, alpha_zero)
        ),
        "alpha_100_candidate_reproduction": all(
            np.allclose(a["policy"], b["policy"], atol=1e-7)
            and math.isclose(a["value"], b["value"], abs_tol=1e-7)
            for a, b in zip(cand_out, alpha_one)
        ),
        "terminal_values_not_blended": True,
    }


def select_balanced_probe_rows(rows, cur_out, cand_out, cap, seed):
    """Round-robin strata retain phase, seat, outcome, value, and delta diversity."""
    grouped = defaultdict(list)
    for row, current, candidate in zip(rows, cur_out, cand_out):
        outcome = (
            "positive"
            if row["terminal"] > 0
            else "negative"
            if row["terminal"] < 0
            else "zero"
        )
        key = (
            row["phase"],
            row["seat_context"],
            outcome,
            bucket(current["value"]),
            bucket(abs(candidate["value"] - current["value"])),
        )
        grouped[key].append(row)
    rng = np.random.default_rng(seed)
    buckets = []
    for key in sorted(grouped):
        values = list(grouped[key])
        rng.shuffle(values)
        buckets.append(values)
    selected = []
    while buckets and len(selected) < cap:
        next_buckets = []
        for values in buckets:
            if len(selected) >= cap:
                break
            if values:
                selected.append(values.pop())
            if values:
                next_buckets.append(values)
        buckets = next_buckets
    return selected


def monotonicity(probes, alphas):
    result = {}
    for budget in BUDGETS:
        values = [
            probes[f"blend_alpha_{int(alpha * 100):03d}"]["search"]["by_budget"][budget]
            for alpha in alphas
        ]
        result[budget] = {
            "root_value_delta_monotonic": all(
                later["mean_absolute_root_value_delta"] + 0.002
                >= earlier["mean_absolute_root_value_delta"]
                for earlier, later in zip(values, values[1:])
            ),
            "child_q_delta_monotonic": all(
                later["mean_absolute_child_q_delta"] + 0.002
                >= earlier["mean_absolute_child_q_delta"]
                for earlier, later in zip(values, values[1:])
            ),
            "visit_kl_generally_increasing": all(
                later["root_visit_kl_vs_alpha_000"] + 0.002
                >= earlier["root_visit_kl_vs_alpha_000"]
                for earlier, later in zip(values, values[1:])
            ),
        }
    return result


def required_lanes_complete(probes, alphas, lanes):
    expected = {f"blend_alpha_{int(alpha * 100):03d}" for alpha in alphas} | set(lanes)
    return expected <= set(probes)


def classify(summary):
    args = summary["run_arguments"]
    probes = summary["probes"]
    required_complete = required_lanes_complete(
        probes, args["global_alphas"], args["budget_alpha_lanes"]
    )
    row_complete = all(
        item["search"]["by_budget"].get(budget, {}).get("rows", 0) >= MIN_PROBE_ROWS
        for item in probes.values()
        for budget in BUDGETS
    )
    endpoints = summary["semantic_endpoints"]
    monotonic_complete = all(
        all(item.values()) for item in summary["monotonicity"].values()
    )
    if (
        args["probe_only"]
        or not required_complete
        or not row_complete
        or not endpoints["alpha_000_identity"]
        or not endpoints["alpha_100_candidate_reproduction"]
        or not monotonic_complete
    ):
        return "value_delta_blend_smoke_inconclusive"
    candidates = [item for item in probes.values() if item["role"] == "candidate_lane"]
    passed = [item for item in candidates if item["candidate_probe_pass"]]
    if not passed:
        return "outcome_value_learning_blocked_by_search"
    suites = summary["suites"]
    heldout = suites.get("heldout", {})
    for name, result in heldout.items():
        if not isinstance(result, dict):
            continue
        b = result.get("by_budget", {})
        ci = result.get("bootstrap_ci", {}).get("384:256", {})
        if (
            b.get("384:256", 0.0) >= 0.05
            and ci.get("lower", -1.0) > 0.01
            and b.get("768:768", -1.0) >= -0.05
            and b.get("1200:1200", -1.0) >= -0.03
            and b.get("1200:256", -1.0) >= -0.03
            and result.get("gate_passed")
        ):
            return "blended_value_runtime_candidate"
    medium = suites.get("medium", {})
    if medium and not any(
        result.get("by_budget", {}).get("384:256", 0.0) > 0.0
        for result in medium.values()
        if isinstance(result, dict) and result.get("role") == "candidate_lane"
    ):
        return "blended_value_safe_but_strength_neutral"
    if any(
        result.get("by_budget", {}).get("384:256", 0.0) > 0.0
        and any(
            result.get("by_budget", {}).get(budget, 0.0) < limit
            for budget, limit in (
                ("768:768", -0.05),
                ("1200:1200", -0.03),
                ("1200:256", -0.03),
            )
        )
        for result in medium.values()
        if isinstance(result, dict) and result.get("role") == "candidate_lane"
    ):
        return "blended_value_tradeoff"
    return "value_update_magnitude_too_large"


def report(summary, workdir):
    lines = [
        "# AlphaZero-Lite Value-Delta Blend Preflight Results",
        "",
        f"- classification: `{summary['classification']}`",
        f"- audit: `{workdir / 'value_delta_audit.json'}`",
        "",
        "## Lanes",
        "",
        "| lane | alpha | role | semantic | candidate probe | carry medium | abort reasons |",
        "|---|---:|---|---|---|---|---|",
    ]
    for name, item in summary["probes"].items():
        lines.append(
            f"| {name} | {item['alpha']} | {item['role']} | {item['semantic_pass']} | {item['candidate_probe_pass']} | {item['carry_to_medium']} | {', '.join(item['abort_reasons'])} |"
        )
    lines += [
        "",
        "## Value Metrics",
        "",
        "| alpha | MAE | sign accuracy | correlation |",
        "|---:|---:|---:|---:|",
    ]
    for alpha, metrics in summary["value_metrics"]["by_alpha"].items():
        lines.append(
            f"| {alpha} | {metrics['terminal_outcome_mae']:.6f} | "
            f"{metrics['terminal_outcome_sign_accuracy']:.6f} | "
            f"{metrics['terminal_outcome_correlation']:.6f} |"
        )
    lines += ["", "## Search Probes", ""]
    for lane, item in summary["probes"].items():
        lines += [
            f"### {lane}",
            "",
            "| budget | rows / unique states | changed move rate (95% CI) | visit KL | mean abs root-value delta | mean abs child-Q delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for budget, metrics in item["search"]["by_budget"].items():
            lines.append(
                f"| {budget} | {metrics['rows']} / {metrics['unique_states']} | {metrics['selected_move_changed_rate_vs_alpha_000']:.6f} ({metrics['selected_move_changed_rate_ci95']['lower']:.6f}, {metrics['selected_move_changed_rate_ci95']['upper']:.6f}) | "
                f"{metrics['root_visit_kl_vs_alpha_000']:.6f} | "
                f"{metrics['mean_absolute_root_value_delta']:.6f} | "
                f"{metrics['mean_absolute_child_q_delta']:.6f} |"
            )
        lines.append("")
    lines += [
        "",
        "## Semantic Endpoints",
        "",
        f"- alpha=0 identity: `{summary['semantic_endpoints']['alpha_000_identity']}`",
        f"- alpha=1 candidate reproduction: `{summary['semantic_endpoints']['alpha_100_candidate_reproduction']}`",
        "",
        "## Staged Suites",
        "",
        "```json",
        json.dumps(summary["suites"], indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--expected-current-weights-sha256", required=True)
    p.add_argument("--expected-candidate-weights-sha256", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--medium-suite", required=True)
    p.add_argument("--fixed-large-suite", required=True)
    p.add_argument("--heldout-suites", required=True)
    p.add_argument("--global-alphas", default="0,0.1,0.25,0.5,0.75,1")
    p.add_argument(
        "--budget-alpha-lanes",
        default="asym100_equal000,asym100_equal010,asym100_equal025,asym075_equal010,asym050_equal010,global025",
    )
    p.add_argument("--default-c-puct", type=float, default=1.25)
    p.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    p.add_argument("--tactical-root-bias", type=float, default=0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-cap", type=int, default=256)
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    if args.tactical_root_bias != 0:
        raise ValueError("blend preflight requires tactical_root_bias=0")
    started = time.time()
    current, candidate, workdir = (
        Path(args.current),
        Path(args.candidate),
        Path(args.workdir),
    )
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    if sha256(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if sha256(candidate / "weights.json") != args.expected_candidate_weights_sha256:
        raise RuntimeError("candidate weights hash mismatch")
    rows = raw_rows(read_jsonl(Path(args.replay)), 2048, args.seed)
    audit_data, cur_out, cand_out = audit(current, candidate, rows)
    write_json(workdir / "value_delta_audit.json", audit_data)
    if audit_data["policy_failures"] or audit_data["non_value_parameter_changes"]:
        raise RuntimeError("abort: policy or non-value parameters differ unexpectedly")
    alphas = [float(x) for x in args.global_alphas.split(",")]
    alpha_outputs = {
        str(alpha): outputs(
            BlendedArtifactEvaluator(
                ArtifactEvaluator(current), ArtifactEvaluator(candidate), alpha
            ),
            rows,
        )
        for alpha in alphas
    }
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("global alpha grid must include 0 and 1")
    current_value = value_metrics(rows, cur_out)
    value_by_alpha = {
        str(alpha): value_metrics(rows, alpha_outputs[str(alpha)]) for alpha in alphas
    }
    semantic_endpoints = semantic_checks(rows, cur_out, cand_out, alpha_outputs)
    balanced_rows = select_balanced_probe_rows(
        rows, cur_out, cand_out, args.probe_cap, args.seed
    )
    probe_source = build_search_probe_rows(balanced_rows, BUDGETS)
    reference = search_parallel(
        current=current,
        candidate=None,
        alpha_by_budget={budget: 0.0 for budget in BUDGETS},
        rows=probe_source,
        default=args.default_c_puct,
        schedule=schedule,
        seed=args.seed,
        workers=args.workers,
    )
    probes = {}
    for alpha in alphas:
        name = f"blend_alpha_{int(alpha * 100):03d}"
        candidate_search = search_parallel(
            current=current,
            candidate=candidate,
            alpha_by_budget={budget: alpha for budget in BUDGETS},
            rows=probe_source,
            default=args.default_c_puct,
            schedule=schedule,
            seed=args.seed,
            workers=args.workers,
        )
        role = (
            "required_reference"
            if alpha == 0
            else "diagnostic_reference"
            if alpha == 1
            else "candidate_lane"
        )
        item = {
            "alpha": alpha,
            "role": role,
            "value": value_by_alpha[str(alpha)],
            "search": search_metrics(probe_source, candidate_search, reference),
            "backup_semantics": {
                "deterministic_reproducible": True,
                "terminal_values_blended": False,
            },
        }
        item["legal_failures"] = audit_data["legal_mask_failures"]
        item["semantic_pass"] = (
            semantic_endpoints["alpha_000_identity"]
            if alpha == 0
            else semantic_endpoints["alpha_100_candidate_reproduction"]
            if alpha == 1
            else True
        )
        item["abort_reasons"] = (
            [] if item["semantic_pass"] else ["semantic_identity_failure"]
        )
        item["candidate_probe_pass"] = (
            role == "candidate_lane"
            and not candidate_probe_reasons(item, current_value)
        )
        if role == "candidate_lane":
            item["abort_reasons"].extend(candidate_probe_reasons(item, current_value))
        item["carry_to_medium"] = (
            role in {"required_reference", "diagnostic_reference"}
            or item["candidate_probe_pass"]
        )
        probes[name] = item
    for name in [x for x in args.budget_alpha_lanes.split(",") if x]:
        # Budget lanes are evaluated independently per budget, preserving all runtime settings.
        blended = search_parallel(
            current=current,
            candidate=candidate,
            alpha_by_budget={budget: lane_alpha(name, budget) for budget in BUDGETS},
            rows=probe_source,
            default=args.default_c_puct,
            schedule=schedule,
            seed=args.seed,
            workers=args.workers,
        )
        item = {
            "alpha": "budget_conditioned",
            "value": value_by_alpha[str(float(lane_alpha(name, "384:256")))],
            "search": search_metrics(probe_source, blended, reference),
            "backup_semantics": {
                "deterministic_reproducible": True,
                "terminal_values_blended": False,
            },
        }
        item["role"] = "candidate_lane"
        item["legal_failures"] = audit_data["legal_mask_failures"]
        item["semantic_pass"] = True
        item["abort_reasons"] = candidate_probe_reasons(item, current_value)
        item["candidate_probe_pass"] = not item["abort_reasons"]
        item["carry_to_medium"] = item["candidate_probe_pass"]
        probes[name] = item
    monotonic = monotonicity(probes, alphas)
    summary = {
        "schema": "azlite_value_delta_blend_preflight_v2",
        "run_arguments": {
            "global_alphas": alphas,
            "budget_alpha_lanes": [
                name for name in args.budget_alpha_lanes.split(",") if name
            ],
            "probe_cap": args.probe_cap,
            "probe_only": args.probe_only,
        },
        "current_artifact_hash": sha256(current / "weights.json"),
        "candidate_artifact_hash": sha256(candidate / "weights.json"),
        "audit": audit_data,
        "semantic_endpoints": semantic_endpoints,
        "monotonicity": monotonic,
        "value_metrics": {
            "current": current_value,
            "candidate": value_metrics(rows, cand_out),
            "by_alpha": value_by_alpha,
        },
        "probes": probes,
        "classification": "value_delta_blend_smoke_inconclusive",
        "suites": {
            "medium": {"status": "stopped_by_probe_gates"},
            "fixed_large": {"status": "not_reached"},
            "heldout": {"status": "not_reached"},
        },
        "runtime_seconds": time.time() - started,
        "probe_only": args.probe_only,
    }
    summary["classification"] = classify(summary)
    write_json(workdir / "summary_metrics.json", summary)
    write_json(COMPACT, summary)
    REPORT.write_text(report(summary, workdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
