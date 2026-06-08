#!/usr/bin/env python3
"""Seat-aware search-compression sweep for iter0_reference.

Evaluates the candidate against current under the seat-aware protocol while
sweeping MCTS/evaluation search settings (c_puct, root_policy_mode,
tactical_root_bias, root_prior_transform). The goal is to determine whether the
high-search breakthrough from iter0_reference can be surfaced at practical or
near-practical search budgets by changing only search parameters.

Does not train, promote, or overwrite any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")

BUDGET_PAIR_LABELS = {
    (384, 256): "standard_384_256",
    (768, 256): "moderate_challenger_768_256",
    (768, 768): "moderate_equal_768_768",
    (1200, 1200): "equal_high_1200_1200",
    (256, 768): "current_high_asymmetry_256_768",
}

SEARCH_LANES = [
    {
        "name": "default_eval",
        "c_puct": 1.25,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "c_puct_low",
        "c_puct": 0.75,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "c_puct_mid_low",
        "c_puct": 1.0,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "c_puct_default",
        "c_puct": 1.25,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "c_puct_high",
        "c_puct": 1.50,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "c_puct_very_high",
        "c_puct": 2.0,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "root_visit_count",
        "c_puct": 1.25,
        "root_policy_mode": "visit_count",
        "tactical_root_bias": 0.1,
        "root_prior_transform": None,
    },
    {
        "name": "tactical_bias_off",
        "c_puct": 1.25,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.0,
        "root_prior_transform": None,
    },
    {
        "name": "tactical_bias_high",
        "c_puct": 1.25,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.2,
        "root_prior_transform": None,
    },
    {
        "name": "root_prior_transform_damp_010",
        "c_puct": 1.25,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
        "root_prior_transform": "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seat-aware search-compression sweep for iter0_reference."
    )
    parser.add_argument("--workdir", required=True, help="Working directory")
    parser.add_argument("--current", required=True, help="Current artifact path")
    parser.add_argument("--candidate", required=True, help="Candidate artifact path")
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,768:768,1200:1200,256:768",
        help="Comma-separated challenger:current simulation pairs",
    )
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--extended-games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated lane names to run (default: all)",
    )
    parser.add_argument(
        "--skip-arenas",
        action="store_true",
        help="Skip arena runs (use cached results)",
    )
    parser.add_argument(
        "--out-report",
        default=None,
        help="Override output report path",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(score: float, sample_size: int) -> dict[str, float]:
    z = 1.96
    if sample_size <= 0:
        return {"lower": 0.0, "upper": 0.0}
    denominator = 1.0 + ((z**2) / sample_size)
    center = score + ((z**2) / (2.0 * sample_size))
    margin = (
        z
        * (((score * (1.0 - score)) + ((z**2) / (4.0 * sample_size))) / sample_size)
        ** 0.5
    )
    return {
        "lower": max(0.0, (center - margin) / denominator),
        "upper": min(1.0, (center + margin) / denominator),
    }


def run_arena_subprocess(
    *,
    challenger: str,
    current: str,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    c_puct: float,
    root_policy_mode: str,
    tactical_root_bias: float,
    root_prior_transform: str | None = None,
    challenger_starts: int | None = None,
    out_json: str,
    out_jsonl: str,
    timeout: int = 3600,
) -> dict:
    cmd = [
        VENV_PYTHON,
        str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        challenger,
        "--current",
        current,
        "--challenger-simulations",
        str(challenger_sims),
        "--current-simulations",
        str(current_sims),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(max(1, workers)),
        "--min-score",
        "0.0",
        "--c-puct",
        str(c_puct),
        "--root-policy-mode",
        root_policy_mode,
        "--tactical-root-bias",
        str(tactical_root_bias),
        "--out",
        out_json,
        "--game-jsonl",
        out_jsonl,
    ]
    if challenger_starts is not None:
        cmd.extend(["--challenger-starts", str(challenger_starts)])
    if root_prior_transform is not None:
        cmd.extend(["--root-prior-transform", root_prior_transform])

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"arena failed ({challenger_sims}:{current_sims}, "
            f"c_puct={c_puct}, mode={root_policy_mode}, bias={tactical_root_bias}): "
            f"{stderr or ' '.join(cmd)}"
        )
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def parse_game_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_seat_split_metrics(entries: list[dict]) -> dict:
    p0_wins = p0_losses = p0_draws = 0
    p1_wins = p1_losses = p1_draws = 0
    first_moves_challenger: Counter = Counter()
    first_moves_current: Counter = Counter()
    p0_margins: list[int] = []
    p1_margins: list[int] = []
    p0_lengths: list[int] = []
    p1_lengths: list[int] = []
    trajectory_hashes: list[str] = []

    for ent in entries:
        cp = ent["challenger_player"]
        winner = ent["winner"]
        margin = ent["margin"]
        if cp == 0:
            if winner == "challenger":
                p0_wins += 1
            elif winner == "current":
                p0_losses += 1
            else:
                p0_draws += 1
            p0_margins.append(margin)
            p0_lengths.append(ent["game_length"])
        else:
            if winner == "challenger":
                p1_wins += 1
            elif winner == "current":
                p1_losses += 1
            else:
                p1_draws += 1
            p1_margins.append(margin)
            p1_lengths.append(ent["game_length"])

        if ent.get("first_move_challenger") is not None:
            first_moves_challenger[str(ent["first_move_challenger"])] += 1
        if ent.get("first_move_current") is not None:
            first_moves_current[str(ent["first_move_current"])] += 1
        trajectory_hashes.append(ent["trajectory"])

    p0_total = p0_wins + p0_losses + p0_draws
    p1_total = p1_wins + p1_losses + p1_draws

    traj_counter = Counter(trajectory_hashes)
    duplicate_count = sum(c for c in traj_counter.values() if c > 1)

    p0_score = (p0_wins + 0.5 * p0_draws) / max(p0_total, 1)
    p1_score = (p1_wins + 0.5 * p1_draws) / max(p1_total, 1)

    return {
        "challenger_starts_0": {
            "games": p0_total,
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "score": p0_score,
            "ci95": wilson_interval(p0_score, p0_total),
            "margin_mean": statistics.fmean(p0_margins) if p0_margins else 0.0,
            "margin_median": statistics.median(p0_margins) if p0_margins else 0.0,
            "game_length_mean": statistics.fmean(p0_lengths) if p0_lengths else 0.0,
            "game_length_median": statistics.median(p0_lengths) if p0_lengths else 0.0,
        },
        "challenger_starts_1": {
            "games": p1_total,
            "wins": p1_wins,
            "losses": p1_losses,
            "draws": p1_draws,
            "score": p1_score,
            "ci95": wilson_interval(p1_score, p1_total),
            "margin_mean": statistics.fmean(p1_margins) if p1_margins else 0.0,
            "margin_median": statistics.median(p1_margins) if p1_margins else 0.0,
            "game_length_mean": statistics.fmean(p1_lengths) if p1_lengths else 0.0,
            "game_length_median": statistics.median(p1_lengths) if p1_lengths else 0.0,
        },
        "disadvantaged_seat_score": p1_score,
        "first_move_challenger_dist": dict(first_moves_challenger.most_common(6)),
        "first_move_current_dist": dict(first_moves_current.most_common(6)),
        "margin_mean": statistics.fmean(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "margin_median": statistics.median(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "game_length_mean": statistics.fmean(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "game_length_median": statistics.median(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "unique_trajectories": len(traj_counter),
        "duplicate_trajectory_count": duplicate_count,
        "total_games": p0_total + p1_total,
    }


def run_lane(
    *,
    lane: dict,
    candidate: str,
    current: str,
    candidate_sha: str,
    current_sha: str,
    budget_pairs: list[tuple[int, int]],
    games: int,
    seed: int,
    workers: int,
    lane_dir: Path,
    skip_arenas: bool,
) -> dict:
    lane_dir.mkdir(parents=True, exist_ok=True)
    budget_results: list[dict] = []

    for chall_sims, curr_sims in budget_pairs:
        budget_label = BUDGET_PAIR_LABELS.get(
            (chall_sims, curr_sims), f"{chall_sims}_vs_{curr_sims}"
        )
        pair_dir = lane_dir / budget_label
        pair_dir.mkdir(parents=True, exist_ok=True)

        alt_json = str(pair_dir / "alternating_arena.json")
        alt_jsonl = str(pair_dir / "alternating_games.jsonl")

        print(f"  [{lane['name']}] {budget_label} alternating ...", flush=True)
        t0 = time.time()
        try:
            if not skip_arenas:
                alt_report = run_arena_subprocess(
                    challenger=candidate,
                    current=current,
                    challenger_sims=chall_sims,
                    current_sims=curr_sims,
                    games=games,
                    seed=seed,
                    workers=workers,
                    c_puct=lane["c_puct"],
                    root_policy_mode=lane["root_policy_mode"],
                    tactical_root_bias=lane["tactical_root_bias"],
                    root_prior_transform=lane.get("root_prior_transform"),
                    out_json=alt_json,
                    out_jsonl=alt_jsonl,
                )
            else:
                alt_report = json.loads(Path(alt_json).read_text(encoding="utf-8"))
        except RuntimeError as e:
            print(f"    FAILED: {e}", flush=True)
            budget_results.append(
                {
                    "budget_label": budget_label,
                    "challenger_simulations": chall_sims,
                    "current_simulations": curr_sims,
                    "error": str(e),
                }
            )
            continue
        elapsed = time.time() - t0

        alt_entries = parse_game_jsonl(alt_jsonl)
        seat_metrics = compute_seat_split_metrics(alt_entries)
        arena_notes = (
            alt_report.get("notes", {})
            if isinstance(alt_report.get("notes"), dict)
            else {}
        )
        move_time_mean_ms = arena_notes.get("move_time_mean_ms")
        move_time_p95_ms = arena_notes.get("move_time_p95_ms")

        result_row = {
            "budget_label": budget_label,
            "challenger_simulations": chall_sims,
            "current_simulations": curr_sims,
            "alternating_score": alt_report["score"],
            "alternating_wins": alt_report["wins"],
            "alternating_losses": alt_report["losses"],
            "alternating_draws": alt_report["draws"],
            "seat_metrics": seat_metrics,
            "move_time_mean_ms": move_time_mean_ms,
            "move_time_p95_ms": move_time_p95_ms,
            "wall_seconds": round(elapsed, 1),
        }
        budget_results.append(result_row)

        print(
            f"    score={alt_report['score']:.4f} "
            f"ds_score={seat_metrics['disadvantaged_seat_score']:.4f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    return {
        "lane_name": lane["name"],
        "c_puct": lane["c_puct"],
        "root_policy_mode": lane["root_policy_mode"],
        "tactical_root_bias": lane["tactical_root_bias"],
        "root_prior_transform": lane.get("root_prior_transform"),
        "candidate_sha256": candidate_sha,
        "current_sha256": current_sha,
        "budget_results": budget_results,
    }


def classify_search_compression(
    lane_result: dict,
    positive_control_ds: float | None,
) -> str:
    """Classify the search-setting lane."""
    disabled_seat_scores: dict[str, float] = {}
    for br in lane_result.get("budget_results", []):
        budget_label = br.get("budget_label", "unknown")
        seat_metrics = br.get("seat_metrics", {})
        if isinstance(seat_metrics, dict):
            disabled_seat_scores[budget_label] = float(
                seat_metrics.get("disadvantaged_seat_score", 0.0)
            )

    ds_384_256 = disabled_seat_scores.get("standard_384_256", 0.0)
    ds_768_256 = disabled_seat_scores.get("moderate_challenger_768_256", 0.0)
    ds_768_768 = disabled_seat_scores.get("moderate_equal_768_768", 0.0)
    ds_1200_1200 = disabled_seat_scores.get("equal_high_1200_1200", 0.0)
    ds_256_768 = disabled_seat_scores.get("current_high_asymmetry_256_768", 0.0)

    lost_1200_breakthrough = (
        positive_control_ds is not None
        and positive_control_ds > 0.1
        and ds_1200_1200 <= 0.1
    )

    if lost_1200_breakthrough:
        return "search_setting_regression"

    collapsed_asymmetry = ds_256_768 >= 0.5

    settings_promising = ds_384_256 > 0.0 or ds_768_256 >= 0.50 or ds_768_768 >= 0.50

    if settings_promising and not collapsed_asymmetry:
        return "search_compression_promising"

    has_high_breakthrough = ds_1200_1200 > 0.1

    if has_high_breakthrough:
        return "high_search_only"

    if ds_1200_1200 <= 0.0 and ds_384_256 <= 0.0 and ds_768_256 <= 0.0:
        return "search_setting_regression"

    return "unclassified"


def build_ranking_table(
    lane_results: list[dict],
    positive_control_ds: float | None,
) -> list[dict]:
    rows = []
    for lr in lane_results:
        disabled_scores: dict[str, float] = {}
        move_time = {}
        for br in lr.get("budget_results", []):
            bl = br.get("budget_label", "unknown")
            sm = br.get("seat_metrics", {})
            if isinstance(sm, dict):
                disabled_scores[bl] = float(sm.get("disadvantaged_seat_score", 0.0))
            move_time[bl] = br.get("move_time_mean_ms")
        classification = classify_search_compression(lr, positive_control_ds)

        p95_values = [
            br.get("move_time_p95_ms")
            for br in lr.get("budget_results", [])
            if br.get("move_time_p95_ms") is not None
        ]
        latency_p95 = max(p95_values) if p95_values else None

        row = {
            "lane_name": lr["lane_name"],
            "c_puct": lr["c_puct"],
            "root_policy_mode": lr["root_policy_mode"],
            "tactical_root_bias": lr["tactical_root_bias"],
            "root_prior_transform": lr.get("root_prior_transform"),
            "ds_384_256": disabled_scores.get("standard_384_256"),
            "ds_768_256": disabled_scores.get("moderate_challenger_768_256"),
            "ds_768_768": disabled_scores.get("moderate_equal_768_768"),
            "ds_1200_1200": disabled_scores.get("equal_high_1200_1200"),
            "ds_256_768": disabled_scores.get("current_high_asymmetry_256_768"),
            "classification": classification,
            "latency_p95_ms": latency_p95,
        }
        rows.append(row)

    def rank_key(r: dict) -> tuple:
        ds384 = float(r.get("ds_384_256") or -1.0)
        ds768_256 = float(r.get("ds_768_256") or -1.0)
        ds768_768 = float(r.get("ds_768_768") or -1.0)
        ds1200 = float(r.get("ds_1200_1200") or -1.0)
        return (-ds384, -ds768_256, -ds768_768, -ds1200)

    return sorted(rows, key=rank_key)


def build_summary_report(
    *,
    candidate: str,
    current: str,
    candidate_sha: str,
    current_sha: str,
    budget_pairs: list[tuple[int, int]],
    games: int,
    seed: int,
    lane_results: list[dict],
    ranking_table: list[dict],
    positive_control_ds: float | None,
) -> dict:
    return {
        "schema": "azlite_search_compression_sweep_v1",
        "candidate_path": candidate,
        "candidate_sha256": candidate_sha,
        "current_path": current,
        "current_sha256": current_sha,
        "budget_pairs": [
            {"challenger_simulations": cs, "current_simulations": cur}
            for cs, cur in budget_pairs
        ],
        "games_per_pair": games,
        "seed": seed,
        "positive_control_disadvantaged_seat_score": positive_control_ds,
        "lane_results": lane_results,
        "ranking_table": ranking_table,
    }


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    candidate_sha = sha256_file(Path(args.candidate) / "weights.json")
    current_sha = sha256_file(Path(args.current) / "weights.json")

    print(f"Candidate SHA256: {candidate_sha}")
    print(f"Current SHA256:   {current_sha}")

    budget_pairs = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c, cur = bp.split(":", 1)
            budget_pairs.append((int(c), int(cur)))

    lanes_to_run = SEARCH_LANES
    if args.lanes is not None:
        requested = set(args.lanes.split(","))
        lanes_to_run = [la for la in SEARCH_LANES if la["name"] in requested]
        if not lanes_to_run:
            print(f"ERROR: no lanes matched requested names: {args.lanes}")
            return 1

    print(f"\nBudget pairs: {budget_pairs}")
    print(f"Lanes: {len(lanes_to_run)}")
    print(f"Games per pair: {args.games}")
    print(f"Seed: {args.seed}")
    print(f"Workers: {args.workers}")
    print()

    lane_results: list[dict] = []
    total_start = time.time()

    for lane in lanes_to_run:
        lane_dir = workdir / lane["name"]
        print(
            f"[lane] {lane['name']} "
            f"(c_puct={lane['c_puct']}, "
            f"policy={lane['root_policy_mode']}, "
            f"bias={lane['tactical_root_bias']}"
            + (
                f", transform={lane['root_prior_transform']}"
                if lane.get("root_prior_transform")
                else ""
            )
            + ")",
            flush=True,
        )

        lr = run_lane(
            lane=lane,
            candidate=args.candidate,
            current=args.current,
            candidate_sha=candidate_sha,
            current_sha=current_sha,
            budget_pairs=budget_pairs,
            games=args.games,
            seed=args.seed,
            workers=args.workers,
            lane_dir=lane_dir,
            skip_arenas=args.skip_arenas,
        )
        lane_results.append(lr)

        for br in lr.get("budget_results", []):
            ds = (
                br.get("seat_metrics", {}).get("disadvantaged_seat_score")
                if isinstance(br.get("seat_metrics"), dict)
                else None
            )
            print(
                f"  {br.get('budget_label', '?'):40s} "
                f"alt={br.get('alternating_score', 0):.4f} "
                f"ds={ds if ds is not None else 'N/A'}",
                flush=True,
            )

    total_elapsed = time.time() - total_start

    positive_control_ds = None
    default_eval_lr = next(
        (lr for lr in lane_results if lr["lane_name"] == "c_puct_default"), None
    )
    if default_eval_lr:
        for br in default_eval_lr.get("budget_results", []):
            if br.get("budget_label") == "equal_high_1200_1200":
                sm = br.get("seat_metrics", {})
                if isinstance(sm, dict):
                    positive_control_ds = float(sm.get("disadvantaged_seat_score", 0.0))

    ranking_table = build_ranking_table(lane_results, positive_control_ds)

    print(f"\n{'=' * 80}")
    print("RANKING TABLE")
    print(f"{'=' * 80}")
    header = (
        f"{'Lane':<32s} {'c_puct':>7s} {'policy':>14s} {'bias':>6s} "
        f"{'DS 384:256':>11s} {'DS 768:256':>11s} {'DS 768:768':>11s} "
        f"{'DS 1200:1200':>12s} {'DS 256:768':>11s} {'Classification':>28s}"
    )
    print(header)
    print("-" * len(header))
    for row in ranking_table:
        ds384 = f"{row['ds_384_256']:.2f}" if row["ds_384_256"] is not None else "-"
        ds768s = f"{row['ds_768_256']:.2f}" if row["ds_768_256"] is not None else "-"
        ds768e = f"{row['ds_768_768']:.2f}" if row["ds_768_768"] is not None else "-"
        ds1200 = (
            f"{row['ds_1200_1200']:.2f}" if row["ds_1200_1200"] is not None else "-"
        )
        ds256a = f"{row['ds_256_768']:.2f}" if row["ds_256_768"] is not None else "-"
        print(
            f"{row['lane_name']:<32s} {row['c_puct']:>7.2f} "
            f"{row['root_policy_mode']:>14s} {row['tactical_root_bias']:>6.2f} "
            f"{ds384:>11s} {ds768s:>11s} {ds768e:>11s} "
            f"{ds1200:>12s} {ds256a:>11s} {row['classification']:<28s}"
        )

    print(f"\nTotal wall time: {total_elapsed:.0f}s")

    report = build_summary_report(
        candidate=args.candidate,
        current=args.current,
        candidate_sha=candidate_sha,
        current_sha=current_sha,
        budget_pairs=budget_pairs,
        games=args.games,
        seed=args.seed,
        lane_results=lane_results,
        ranking_table=ranking_table,
        positive_control_ds=positive_control_ds,
    )

    out_path = Path(args.out_report) if args.out_report else workdir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
