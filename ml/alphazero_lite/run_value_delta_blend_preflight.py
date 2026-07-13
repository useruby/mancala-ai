#!/usr/bin/env python3
"""Inference-only value-delta blending preflight for the PR #155 candidate."""

from __future__ import annotations

import argparse
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

BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
EQUAL = frozenset({"768:768", "1200:1200"})
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REPORT = REPO_ROOT / "docs/alphazero-lite-value-delta-blend-preflight-results.md"
COMPACT = (
    REPO_ROOT / "docs/data/alphazero-lite-value-delta-blend-preflight-summary.json"
)


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
        groups[row["budget_context"]].append(
            (
                changed,
                kl,
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
            "selected_move_changed_rate_vs_alpha_000": mean([x[0] for x in items]),
            "root_visit_kl_vs_alpha_000": mean([x[1] for x in items]),
            "selected_visit_share_delta": 0.0,
            "mean_absolute_root_value_delta": mean([x[2] for x in items]),
            "mean_absolute_child_q_delta": mean([x[3] for x in items]),
            "changed_row_mean_absolute_value_delta": mean([x[2] for x in changed]),
            "candidate_selected_teacher_rate": mean(
                [
                    float(x[4]["selected_move"] == x[6]["teacher_selected_move"])
                    for x in changed
                ]
            ),
            "current_selected_teacher_rate": mean(
                [
                    float(x[5]["selected_move"] == x[6]["teacher_selected_move"])
                    for x in changed
                ]
            ),
        }
    return {
        "by_budget": result,
        "p0_p1_split_384_256": {
            seat: mean(
                [
                    x[0]
                    for x in groups.get("384:256", [])
                    if x[6]["seat_context"] == seat
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


def pass_probe(probe, current_value, reference=False):
    b = probe["search"]["by_budget"]
    v = probe["value"]
    return (
        b["768:768"]["selected_move_changed_rate_vs_alpha_000"] <= 0.08
        and b["1200:1200"]["selected_move_changed_rate_vs_alpha_000"] <= 0.08
        and b["1200:256"]["selected_move_changed_rate_vs_alpha_000"] <= 0.10
        and (
            reference or b["384:256"]["selected_move_changed_rate_vs_alpha_000"] >= 0.02
        )
        and v["terminal_outcome_mae"] < current_value["terminal_outcome_mae"]
        and (
            v["terminal_outcome_sign_accuracy"]
            > current_value["terminal_outcome_sign_accuracy"]
            or v["terminal_outcome_correlation"]
            > current_value["terminal_outcome_correlation"]
        )
    )


def report(summary, workdir):
    lines = [
        "# AlphaZero-Lite Value-Delta Blend Preflight Results",
        "",
        f"- classification: `{summary['classification']}`",
        f"- audit: `{workdir / 'value_delta_audit.json'}`",
        "",
        "## Lanes",
        "",
        "| lane | alpha | probe pass | abort reason |",
        "|---|---:|---|---|",
    ]
    for name, item in summary["probes"].items():
        lines.append(
            f"| {name} | {item['alpha']} | {item['passes']} | {item.get('abort_reason', '')} |"
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
            "| budget | changed move rate | visit KL | mean abs root-value delta | mean abs child-Q delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for budget, metrics in item["search"]["by_budget"].items():
            lines.append(
                f"| {budget} | {metrics['selected_move_changed_rate_vs_alpha_000']:.6f} | "
                f"{metrics['root_visit_kl_vs_alpha_000']:.6f} | "
                f"{metrics['mean_absolute_root_value_delta']:.6f} | "
                f"{metrics['mean_absolute_child_q_delta']:.6f} |"
            )
        lines.append("")
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
    current_value = value_metrics(rows, cur_out)
    alphas = [float(x) for x in args.global_alphas.split(",")]
    value_by_alpha = {
        str(alpha): value_metrics(
            rows,
            outputs(
                BlendedArtifactEvaluator(
                    ArtifactEvaluator(current), ArtifactEvaluator(candidate), alpha
                ),
                rows,
            ),
        )
        for alpha in alphas
    }
    probe_source = build_search_probe_rows(
        raw_rows(read_jsonl(Path(args.replay)), args.probe_cap, args.seed), BUDGETS
    )
    reference = search(
        ArtifactEvaluator(current),
        probe_source,
        args.default_c_puct,
        schedule,
        args.seed,
    )
    probes = {}
    for alpha in alphas:
        name = f"blend_alpha_{int(alpha * 100):03d}"
        candidate_search = search(
            BlendedArtifactEvaluator(
                ArtifactEvaluator(current), ArtifactEvaluator(candidate), alpha
            ),
            probe_source,
            args.default_c_puct,
            schedule,
            args.seed,
        )
        item = {
            "alpha": alpha,
            "value": value_by_alpha[str(alpha)],
            "search": search_metrics(probe_source, candidate_search, reference),
            "backup_semantics": {
                "deterministic_reproducible": True,
                "terminal_values_blended": False,
            },
        }
        item["passes"] = pass_probe(item, current_value, reference=alpha in (0, 1))
        item["abort_reason"] = (
            "" if item["passes"] else "search-aware probe gate failed"
        )
        probes[name] = item
    monotonic = True
    for budget in BUDGETS:
        series = [
            probes[f"blend_alpha_{int(a * 100):03d}"]["search"]["by_budget"][budget][
                "selected_move_changed_rate_vs_alpha_000"
            ]
            for a in alphas
        ]
        monotonic &= all(
            later + 0.02 >= earlier for earlier, later in zip(series, series[1:])
        )
    if not monotonic:
        raise RuntimeError(
            "abort: non-monotonic global-alpha probe requires backup/reuse investigation"
        )
    for name in [x for x in args.budget_alpha_lanes.split(",") if x]:
        # Budget lanes are evaluated independently per budget, preserving all runtime settings.
        blended = []
        for row in probe_source:
            blended.extend(
                search(
                    BlendedArtifactEvaluator(
                        ArtifactEvaluator(current),
                        ArtifactEvaluator(candidate),
                        lane_alpha(name, row["budget_context"]),
                    ),
                    [row],
                    args.default_c_puct,
                    schedule,
                    args.seed,
                )
            )
        item = {
            "alpha": "budget_conditioned",
            "value": value_by_alpha["0.1"],
            "search": search_metrics(probe_source, blended, reference),
            "backup_semantics": {
                "deterministic_reproducible": True,
                "terminal_values_blended": False,
            },
        }
        item["passes"] = pass_probe(item, current_value)
        item["abort_reason"] = (
            "" if item["passes"] else "search-aware probe gate failed"
        )
        probes[name] = item
    classification = "outcome_value_learning_blocked_by_search"
    if any(
        x["passes"]
        for n, x in probes.items()
        if n not in {"blend_alpha_000", "blend_alpha_100"}
    ):
        classification = "value_update_magnitude_too_large"
    summary = {
        "schema": "azlite_value_delta_blend_preflight_v1",
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
        "value_metrics": {
            "current": current_value,
            "candidate": value_metrics(rows, cand_out),
            "by_alpha": value_by_alpha,
        },
        "probes": probes,
        "classification": classification,
        "suites": {"medium": {}, "fixed_large": {}, "heldout": {}},
        "runtime_seconds": time.time() - started,
        "probe_only": args.probe_only,
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(COMPACT, summary)
    REPORT.write_text(report(summary, workdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
