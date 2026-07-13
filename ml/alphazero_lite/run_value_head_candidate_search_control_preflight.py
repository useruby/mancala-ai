#!/usr/bin/env python3
# ruff: noqa: E402
"""Budget-conditioned runtime-control preflight for PR #155's value candidate.

This runner is deliberately inference-only.  Lane settings are applied only to
the challenger; the current artifact always uses the promoted default profile.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import bootstrap_ci  # noqa: E402
from ml.alphazero_lite.run_current_init_policy_only_distillation_preflight import (  # noqa: E402
    evaluate_policy_probe_metrics,
    evaluate_raw_outputs,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import compute_seat_metrics  # noqa: E402
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    build_search_probe_rows,
    evaluate_search_outputs,
    masked_policy,
    safe_correlation,
    sample_probe_rows,
    split_replay_rows,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402

BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
ASYMMETRIC = frozenset({"384:256", "768:256", "1200:256", "256:768"})
EQUAL = frozenset({"768:768", "1200:1200"})
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-value-head-candidate-search-control-preflight-results.md"
)


@dataclass(frozen=True)
class Lane:
    name: str
    normalize: frozenset[str] = frozenset()
    trust: float | None = None
    cpuct: tuple[tuple[str, float], ...] = ()
    required: bool = False


LANES = {
    "value_head_e2_default": Lane("value_head_e2_default", required=True),
    "normalize_all": Lane("value_head_e2_normalize_all", frozenset(BUDGETS)),
    "normalize_asymmetric_only": Lane(
        "value_head_e2_normalize_asymmetric_only", ASYMMETRIC
    ),
    "value_trust_half_equal_only": Lane(
        "value_head_e2_value_trust_half_equal_only", trust=0.5
    ),
    "value_trust_075_equal_only": Lane(
        "value_head_e2_value_trust_075_equal_only", trust=0.75
    ),
    "cpuct_equal_1_25": Lane(
        "value_head_e2_cpuct_equal_1_25", cpuct=(("768:768", 1.25),)
    ),
    "cpuct_equal_1_50": Lane(
        "value_head_e2_cpuct_equal_1_50", cpuct=(("768:768", 1.5), ("1200:1200", 1.5))
    ),
    "normalize_asym_value_trust_half_equal": Lane(
        "value_head_e2_normalize_asym_value_trust_half_equal", ASYMMETRIC, 0.5
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def lane_options(lane: Lane, budget: str) -> dict[str, Any]:
    options: dict[str, Any] = {"normalize_values": budget in lane.normalize}
    if lane.trust is not None and budget in EQUAL:
        options["value_trust_schedule"] = {
            "enabled": True,
            "opening": lane.trust,
            "midgame": lane.trust,
            "late": lane.trust,
        }
    return options


def lane_cpuct(
    lane: Lane, budget: str, default: float, schedule: dict[str, float]
) -> float:
    overrides = dict(lane.cpuct)
    return float(
        overrides.get(
            budget,
            resolve_budget_cpuct(
                schedule=schedule,
                challenger_simulations=int(budget.split(":")[0]),
                current_simulations=int(budget.split(":")[1]),
                default_c_puct=default,
            ),
        )
    )


def raw_probe_rows(
    replay: list[dict[str, Any]], cap: int = 2048
) -> list[dict[str, Any]]:
    rows = replay[:cap]
    return [
        {
            "raw_state": row["raw_state"],
            "legal_mask": row["legal_mask"],
            "teacher_puct_policy": row["policy"],
            "policy": row["policy"],
            "selected_move": int(row["selected_move"]),
            "budget_context": "replay",
            "phase": row.get("phase", ""),
            "seat_context": row.get("seat_context", ""),
            "terminal": float(row.get("final_outcome", row.get("value", 0.0))),
        }
        for row in rows
    ]


def value_metrics(
    rows: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> dict[str, float]:
    values = [float(item["value"]) for item in outputs]
    terminal = [float(row["terminal"]) for row in rows]
    return {
        "value_mae_vs_terminal_outcome": mean(
            [abs(a - b) for a, b in zip(values, terminal, strict=True)]
        ),
        "value_sign_accuracy": mean(
            [
                1.0 if (a > 0) == (b > 0) and (a < 0) == (b < 0) else 0.0
                for a, b in zip(values, terminal, strict=True)
            ]
        ),
        "root_value_correlation": safe_correlation(values, terminal),
    }


def search_outputs(
    artifact: Path,
    rows: list[dict[str, Any]],
    lane: Lane,
    default: float,
    schedule: dict[str, float],
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    # Existing parallel helper is appropriate for the default lane only.
    if lane.name == "value_head_e2_default":
        return evaluate_search_outputs(
            artifact_path=artifact,
            rows=rows,
            default_c_puct=default,
            cpuct_schedule=schedule,
            tactical_root_bias=0.0,
            seed=seed,
            workers=workers,
        )
    if workers > 1 and len(rows) > 1:
        worker_count = min(workers, len(rows))
        chunk_size = math.ceil(len(rows) / worker_count)
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    search_outputs_chunk,
                    artifact,
                    rows[offset : offset + chunk_size],
                    offset,
                    lane,
                    default,
                    schedule,
                    seed,
                )
                for offset in range(0, len(rows), chunk_size)
            ]
            return [output for future in futures for output in future.result()]
    return search_outputs_chunk(artifact, rows, 0, lane, default, schedule, seed)


def search_outputs_chunk(
    artifact: Path,
    rows: list[dict[str, Any]],
    offset: int,
    lane: Lane,
    default: float,
    schedule: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate a contiguous probe shard in a standalone worker process."""
    evaluator = ArtifactEvaluator(artifact)
    outputs = []
    for index, row in enumerate(rows):
        budget = str(row["budget_context"])
        result = evaluate_artifact_position(
            evaluator=evaluator,
            state=row["raw_state"],
            simulations=int(row["challenger_simulations"]),
            seed=seed + offset + index,
            c_puct=lane_cpuct(lane, budget, default, schedule),
            search_options=build_eval_search_options(
                root_policy_mode="deterministic",
                tactical_root_bias=0.0,
                **lane_options(lane, budget),
            ),
        )
        legal = [move for move, flag in enumerate(row["legal_mask"]) if flag]
        policy = masked_policy(list(result["visits"]), legal)
        selected = int(result["selected_move"])
        outputs.append(
            {
                "selected_move": selected,
                "search_policy": policy,
                "selected_visit_share": policy[selected],
                "root_value": float(
                    result.get("search_root_value", result.get("value", 0.0))
                ),
                "selection_breakdown": result.get("selection_breakdown", {}),
            }
        )
    return outputs


def search_metrics(
    rows: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> dict[str, Any]:
    by_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, cand, cur in zip(rows, candidate, current, strict=True):
        legal = [move for move, flag in enumerate(row["legal_mask"]) if flag]
        changed = int(cand["selected_move"] != cur["selected_move"])
        kl = float(
            sum(
                max(cur["search_policy"][m], 1e-8)
                * math.log(
                    max(cur["search_policy"][m], 1e-8)
                    / max(cand["search_policy"][m], 1e-8)
                )
                for m in legal
            )
        )
        score = 0.0
        for entry in cand.get("selection_breakdown", {}).get("moves") or []:
            if int(entry.get("move", -1)) == cand["selected_move"]:
                score = float(entry.get("selection_score", 0.0))
                break
        by_budget[str(row["budget_context"])].append(
            {
                "changed": changed,
                "kl": kl,
                "share": cand["selected_visit_share"] - cur["selected_visit_share"],
                "dv": abs(cand["root_value"] - cur["root_value"]),
                "score": abs(score),
                "ct": int(cand["selected_move"] == row["teacher_selected_move"]),
                "rt": int(cur["selected_move"] == row["teacher_selected_move"]),
                "seat": row["seat_context"],
            }
        )
    result: dict[str, Any] = {"by_budget": {}}
    for budget, items in by_budget.items():
        changed = [item for item in items if item["changed"]]
        result["by_budget"][budget] = {
            "rows": len(items),
            "changed_rate_vs_current": mean([x["changed"] for x in items]),
            "root_visit_kl_vs_current_ref": mean([x["kl"] for x in items]),
            "selected_visit_share_delta": mean([x["share"] for x in items]),
            "changed_row_mean_abs_dv": mean([x["dv"] for x in changed]),
            "changed_row_mean_abs_dscore": mean([x["score"] for x in changed]),
            "candidate_selected_teacher_rate": mean([x["ct"] for x in changed]),
            "current_selected_teacher_rate": mean([x["rt"] for x in changed]),
        }
    standard = by_budget.get("384:256", [])
    result["p0_p1_split_384_256"] = {
        seat: mean([x["changed"] for x in standard if x["seat"] == seat])
        for seat in ("player_0", "player_1")
    }
    return result


def probe_pass(
    probe: dict[str, Any], current_values: dict[str, float], lane: Lane
) -> bool:
    budgets = probe["search"]["by_budget"]
    return (
        probe["raw"]["legal_failures"] == 0
        and budgets["768:768"]["changed_rate_vs_current"] <= 0.08
        and budgets["1200:1200"]["changed_rate_vs_current"] <= 0.08
        and budgets["1200:256"]["changed_rate_vs_current"] <= 0.10
        and (budgets["384:256"]["changed_rate_vs_current"] >= 0.03 or lane.required)
        and probe["value"]["value_sign_accuracy"]
        > current_values["value_sign_accuracy"]
        and probe["value"]["root_value_correlation"]
        > current_values["root_value_correlation"]
        and probe["raw"]["candidate_entropy"] > 0.05
    )


def run_suite(
    workdir: Path,
    suite: Path,
    current: Path,
    candidate: Path,
    lane: Lane,
    default: float,
    schedule: dict[str, float],
    seed: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    entries = read_jsonl(suite)
    results: dict[str, Any] = {}
    for budget in BUDGETS:
        challenger_sims, current_sims = budget.split(":")
        outdir = workdir / lane.name / budget.replace(":", "_")
        outdir.mkdir(parents=True, exist_ok=True)
        game_entries: list[dict[str, Any]] = []
        candidate_options = json.dumps(lane_options(lane, budget), sort_keys=True)
        current_options = json.dumps({}, sort_keys=True)
        for seat in (0, 1):
            prefix = outdir / f"starts_{seat}_openings.jsonl"
            prefix.write_text(
                "".join(
                    json.dumps({"prefix_moves": row["prefix_moves"]}) + "\n"
                    for row in entries
                ),
                encoding="utf-8",
            )
            games = outdir / f"starts_{seat}_games.jsonl"
            report = outdir / f"starts_{seat}.json"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                str(candidate),
                "--current",
                str(current),
                "--challenger-simulations",
                challenger_sims,
                "--current-simulations",
                current_sims,
                "--games",
                str(len(entries) * 2),
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(prefix),
                "--challenger-starts",
                str(seat),
                "--seed",
                str(seed),
                "--workers",
                str(workers),
                "--c-puct",
                str(
                    resolve_budget_cpuct(
                        schedule=schedule,
                        challenger_simulations=int(challenger_sims),
                        current_simulations=int(current_sims),
                        default_c_puct=default,
                    )
                ),
                "--challenger-c-puct",
                str(lane_cpuct(lane, budget, default, schedule)),
                "--current-c-puct",
                str(
                    resolve_budget_cpuct(
                        schedule=schedule,
                        challenger_simulations=int(challenger_sims),
                        current_simulations=int(current_sims),
                        default_c_puct=default,
                    )
                ),
                "--challenger-search-options-json",
                candidate_options,
                "--current-search-options-json",
                current_options,
                "--root-policy-mode",
                "deterministic",
                "--tactical-root-bias",
                "0.0",
                "--min-score",
                "0",
                "--game-jsonl",
                str(games),
                "--out",
                str(report),
            ]
            completed = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr[-2000:])
            game_entries.extend(read_jsonl(games))
        results[budget] = compute_seat_metrics(game_entries)
    return results


def ds_delta(candidate: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {
        budget: float(candidate[budget]["ds"]) - float(current[budget]["ds"])
        for budget in BUDGETS
    }


def build_report(summary: dict[str, Any], workdir: Path) -> str:
    """Keep the committed report readable while retaining all row-level JSON."""
    lines = [
        "# AlphaZero-Lite Value-Head Candidate Search-Control Preflight Results",
        "",
        "## Artifact Hashes",
        "",
        f"- current artifact hash: `{summary['current_artifact_hash']}`",
        f"- value_head_only_e2 artifact hash: `{summary['candidate_artifact_hash']}`",
        "",
        "## PR #155 Reproduction Table",
        "",
        f"- reproduction JSON: `{workdir / 'pr155_reproduction.json'}`",
        "",
        "## Lane Definitions And Effective Runtime Profiles",
        "",
        "- challenger-only profiles and per-budget c_puct values are in `probes.*.runtime_profile`.",
        "",
        "## Search-Control Support Matrix",
        "",
        "- `normalize_values`, value-trust schedules, and per-side c_puct are supported; no delta clipping lane was added.",
        "",
        "## Search-Aware Probe Table By Budget",
        "",
        "- `probes.*.search.by_budget` records change rate, visit KL, and visit-share deltas.",
        "",
        "## 768:768 Changed-Case Diagnostic Table",
        "",
        "- `probes.*.search.by_budget.768:768` records changed-row |dV|, |dScore|, and teacher rates.",
        "",
        "## Medium DS Table",
        "",
        "- `suites.medium` uses candidate-minus-current DS orientation.",
        "",
        "## Fixed-Large DS Table",
        "",
        "- `suites.fixed_large` uses candidate-minus-current DS orientation.",
        "",
        "## Held-Out Mean/Worst-Suite Table",
        "",
        "- `heldout` contains mean and individual-suite deltas when fixed-large gates pass.",
        "",
        "## Bootstrap CIs",
        "",
        "- orientation: `candidate_minus_current`.",
        "",
        "## P0/P1 Split For 384:256",
        "",
        "- `probes.*.search.p0_p1_split_384_256` records the selected-move change split.",
        "",
        "## Duplicate Trajectory Count",
        "",
        "- per-budget counts are retained in each suite result.",
        "",
        "## Runtime Cost",
        "",
        f"- elapsed seconds: `{summary['runtime_seconds']:.2f}`",
        "",
        "## Gate Result",
        "",
        "- gate run: `False` (this runner does not promote).",
        "",
        "## Final Classification",
        "",
        f"- result: `{summary['classification']}`",
        "",
        f"- complete machine-readable metrics: `{workdir / 'summary_metrics.json'}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--expected-candidate-weights-sha256", required=True)
    parser.add_argument("--pr155-summary", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--lanes", default=",".join(LANES))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--probe-cap", type=int, default=1024)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    workdir = Path(args.workdir)
    current = Path(args.current)
    candidate = Path(args.candidate)
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    if sha256(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if sha256(candidate / "weights.json") != args.expected_candidate_weights_sha256:
        raise RuntimeError("candidate weights hash mismatch")
    requested = [item.strip() for item in args.lanes.split(",") if item.strip()]
    if unknown := set(requested) - set(LANES):
        raise ValueError(f"unknown lanes: {sorted(unknown)}")
    replay = read_jsonl(Path(args.replay))
    _, heldout_rows = split_replay_rows(replay)
    value_rows = raw_probe_rows(
        sample_probe_rows(heldout_rows or replay, seed=args.seed + 1, cap=2048)
    )
    raw_rows = value_rows
    search_rows = build_search_probe_rows(
        raw_probe_rows(
            sample_probe_rows(
                heldout_rows or replay, seed=args.seed, cap=args.probe_cap
            )
        ),
        BUDGETS,
    )
    current_raw = evaluate_raw_outputs(artifact_path=current, rows=raw_rows)["outputs"]
    candidate_raw = evaluate_raw_outputs(artifact_path=candidate, rows=raw_rows)[
        "outputs"
    ]
    current_value = value_metrics(raw_rows, current_raw)
    candidate_value = value_metrics(raw_rows, candidate_raw)
    reproduction = {
        "current_hash": sha256(current / "weights.json"),
        "candidate_hash": sha256(candidate / "weights.json"),
        "pr155_probe_table": {
            "current_ref": current_value,
            "value_head_only_e2": candidate_value,
        },
        "source_summary": str(args.pr155_summary),
    }
    # The published sign/correlation values are rounded; use a deliberately narrow tolerance.
    source = read_json(Path(args.pr155_summary))
    source_probe = source.get("candidate_rows", {})
    reproduction["materially_disagrees"] = (
        not bool(source_probe)
        or abs(candidate_value["value_sign_accuracy"] - 0.6792) > 0.03
        or abs(candidate_value["root_value_correlation"] - 0.5631) > 0.04
    )
    write_json(workdir / "pr155_reproduction.json", reproduction)
    if reproduction["materially_disagrees"]:
        raise RuntimeError("PR #155 reproduction disagrees materially")
    current_search = search_outputs(
        current,
        search_rows,
        Lane("current_ref"),
        args.default_c_puct,
        schedule,
        args.seed,
        args.workers,
    )
    probes: dict[str, Any] = {}
    passing: list[str] = []
    for key in requested:
        lane = LANES[key]
        cand_search = search_outputs(
            candidate,
            search_rows,
            lane,
            args.default_c_puct,
            schedule,
            args.seed,
            args.workers,
        )
        raw = evaluate_policy_probe_metrics(
            rows=raw_rows, candidate_outputs=candidate_raw, current_outputs=current_raw
        )
        probe = {
            "raw": raw,
            "value": candidate_value,
            "search": search_metrics(search_rows, cand_search, current_search),
            "runtime_profile": {
                budget: {
                    "candidate_options": lane_options(lane, budget),
                    "candidate_c_puct": lane_cpuct(
                        lane, budget, args.default_c_puct, schedule
                    ),
                    "current_c_puct": resolve_budget_cpuct(
                        schedule=schedule,
                        challenger_simulations=int(budget.split(":")[0]),
                        current_simulations=int(budget.split(":")[1]),
                        default_c_puct=args.default_c_puct,
                    ),
                }
                for budget in BUDGETS
            },
        }
        probe["passes"] = probe_pass(probe, current_value, lane)
        probes[lane.name] = probe
        if probe["passes"]:
            passing.append(key)
    if args.probe_only:
        write_json(
            workdir / "summary_metrics.json",
            {
                "schema": "azlite_value_head_candidate_search_control_preflight_v1",
                "current_artifact_hash": reproduction["current_hash"],
                "candidate_artifact_hash": reproduction["candidate_hash"],
                "reproduction": reproduction,
                "probes": probes,
                "probe_only": True,
                "runtime_seconds": time.time() - started,
            },
        )
        return 0
    medium_lanes = ["value_head_e2_default", *passing]
    medium_lanes = list(dict.fromkeys(medium_lanes))
    suites: dict[str, Any] = {}
    medium_current = run_suite(
        workdir / "medium" / "current",
        Path(args.medium_suite),
        current,
        current,
        Lane("current_ref"),
        args.default_c_puct,
        schedule,
        args.seed,
        args.workers,
        args.timeout,
    )
    suites["medium"] = {"current": medium_current, "lanes": {}}
    fixed_lanes: list[str] = []
    for key in medium_lanes:
        lane_results = run_suite(
            workdir / "medium",
            Path(args.medium_suite),
            current,
            candidate,
            LANES[key],
            args.default_c_puct,
            schedule,
            args.seed,
            args.workers,
            args.timeout,
        )
        delta = ds_delta(lane_results, medium_current)
        suites["medium"]["lanes"][LANES[key].name] = {
            "results": lane_results,
            "candidate_minus_current": delta,
        }
        if (
            delta["384:256"] >= 0.03
            and delta["768:768"] >= -0.03
            and delta["1200:1200"] >= -0.03
            and delta["1200:256"] >= -0.03
        ):
            fixed_lanes.append(key)
    fixed_lanes = ["value_head_e2_default", *fixed_lanes[:4]]
    fixed_lanes = list(dict.fromkeys(fixed_lanes))
    fixed_current = run_suite(
        workdir / "fixed" / "current",
        Path(args.fixed_large_suite),
        current,
        current,
        Lane("current_ref"),
        args.default_c_puct,
        schedule,
        args.seed,
        args.workers,
        args.timeout,
    )
    suites["fixed_large"] = {"current": fixed_current, "lanes": {}}
    heldout_lanes: list[str] = []
    for key in fixed_lanes:
        result = run_suite(
            workdir / "fixed",
            Path(args.fixed_large_suite),
            current,
            candidate,
            LANES[key],
            args.default_c_puct,
            schedule,
            args.seed,
            args.workers,
            args.timeout,
        )
        delta = ds_delta(result, fixed_current)
        suites["fixed_large"]["lanes"][LANES[key].name] = {
            "results": result,
            "candidate_minus_current": delta,
        }
        if (
            delta["384:256"] >= 0.05
            and delta["768:768"] >= -0.05
            and delta["1200:1200"] >= -0.03
            and delta["1200:256"] >= -0.03
        ):
            heldout_lanes.append(key)
    heldout: dict[str, Any] = {}
    for key in heldout_lanes:
        lane_suites = {}
        diffs: list[float] = []
        for suite_text in args.heldout_suites.split(","):
            suite = Path(suite_text)
            cur = run_suite(
                workdir / "heldout" / "current" / suite.stem,
                suite,
                current,
                current,
                Lane("current_ref"),
                args.default_c_puct,
                schedule,
                args.seed,
                args.workers,
                args.timeout,
            )
            result = run_suite(
                workdir / "heldout" / LANES[key].name / suite.stem,
                suite,
                current,
                candidate,
                LANES[key],
                args.default_c_puct,
                schedule,
                args.seed,
                args.workers,
                args.timeout,
            )
            delta = ds_delta(result, cur)
            lane_suites[suite.stem] = {"candidate_minus_current": delta}
            diffs.append(delta["384:256"])
        heldout[LANES[key].name] = {
            "suites": lane_suites,
            "mean": {
                budget: mean(
                    [
                        item["candidate_minus_current"][budget]
                        for item in lane_suites.values()
                    ]
                )
                for budget in BUDGETS
            },
            "bootstrap_ci_384_256_candidate_minus_current": bootstrap_ci(
                diffs, seed=args.seed
            ),
        }
    classification = "equal_budget_value_sensitivity_uncontrolled"
    if any(
        item["mean"]["384:256"] >= 0.05
        and item["bootstrap_ci_384_256_candidate_minus_current"]["lower"] > 0.01
        and item["mean"]["768:768"] >= -0.05
        and item["mean"]["1200:1200"] >= -0.03
        and item["mean"]["1200:256"] >= -0.03
        for item in heldout.values()
    ):
        classification = "value_head_runtime_candidate"
    elif heldout_lanes:
        classification = "value_head_search_control_promising"
    elif any(
        "normalize" in name
        and suites["fixed_large"]["lanes"][name]["candidate_minus_current"]["384:256"]
        > 0
        and (
            suites["fixed_large"]["lanes"][name]["candidate_minus_current"]["768:768"]
            < -0.03
            or suites["fixed_large"]["lanes"][name]["candidate_minus_current"][
                "1200:1200"
            ]
            < -0.03
        )
        for name in suites["fixed_large"]["lanes"]
    ):
        classification = "normalize_values_tradeoff_confirmed"
    summary = {
        "schema": "azlite_value_head_candidate_search_control_preflight_v1",
        "current_artifact_hash": reproduction["current_hash"],
        "candidate_artifact_hash": reproduction["candidate_hash"],
        "reproduction": reproduction,
        "lanes": {
            LANES[key].name: {
                "normalize_budgets": sorted(LANES[key].normalize),
                "equal_value_trust": LANES[key].trust,
                "cpuct_overrides": dict(LANES[key].cpuct),
            }
            for key in requested
        },
        "probes": probes,
        "suites": suites,
        "heldout": heldout,
        "classification": classification,
        "runtime_seconds": time.time() - started,
    }
    write_json(workdir / "summary_metrics.json", summary)
    REPORT_PATH.write_text(build_report(summary, workdir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
