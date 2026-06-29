#!/usr/bin/env python3
"""Opening-suite seat-aware benchmark for AlphaZero-lite.

Evaluates multiple candidate checkpoints against current across a
deduplicated, diverse opening-prefix suite with forced-seat splits.
Produces per-candidate metrics, per-opening rankings, and a final
candidate ranking table.

Does not train, promote, or overwrite any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    default_runtime_schedule_json,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)


def default_eval_tactical_root_bias() -> float:
    from ml.alphazero_lite.self_play import DEFAULT_EVAL_SEARCH_OPTIONS

    return float(DEFAULT_EVAL_SEARCH_OPTIONS["tactical_root_bias"])


def _find_python() -> str:
    candidates = [
        REPO_ROOT / ".venv/bin/python",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_suite(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_arena(
    *,
    challenger: str,
    current: str,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    out_json: str,
    out_jsonl: str,
    opening_prefixes_jsonl: str,
    challenger_starts: int,
    games_per_opening: int = 1,
    root_policy_mode: str = "deterministic",
    root_temperature: float = 0.0,
    c_puct: float = 1.25,
    tactical_root_bias: float | None = None,
    timeout: int = 7200,
) -> dict:
    python = _find_python()
    cmd = [
        python,
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
        str(workers),
        "--min-score",
        "0.0",
        "--out",
        out_json,
        "--game-jsonl",
        out_jsonl,
        "--challenger-starts",
        str(challenger_starts),
        "--games-per-opening",
        str(games_per_opening),
        "--opening-prefixes-jsonl",
        opening_prefixes_jsonl,
        "--root-policy-mode",
        root_policy_mode,
        "--root-temperature",
        str(root_temperature),
        "--c-puct",
        str(c_puct),
    ]
    if tactical_root_bias is not None:
        cmd.extend(["--tactical-root-bias", str(tactical_root_bias)])

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"arena failed ({challenger_sims}:{current_sims}): {stderr or ' '.join(cmd)}"
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


def cache_matches(cached: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(cached.get(key) == value for key, value in expected.items())


def budget_cache_context(
    *,
    suite_path: str,
    suite_sha: str,
    suite_size: int,
    current_path: str,
    current_sha: str,
    candidate_path: str,
    candidate_sha: str,
    challenger_sims: int,
    current_sims: int,
    games_per_opening: int,
    total_games: int,
    root_policy_mode: str,
    root_temperature: float,
    c_puct: float,
    c_puct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "suite_path": suite_path,
        "suite_sha256": suite_sha,
        "suite_size": suite_size,
        "current_path": current_path,
        "current_sha256": current_sha,
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_sha,
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "games_per_opening": games_per_opening,
        "total_games": total_games,
        "root_policy_mode": root_policy_mode,
        "root_temperature": root_temperature,
        "c_puct": c_puct,
        "c_puct_schedule": c_puct_schedule,
        "tactical_root_bias": tactical_root_bias,
        "seed": seed,
    }


def seat_cache_context(
    budget_context: dict[str, Any], *, challenger_starts: int
) -> dict[str, Any]:
    return {
        **budget_context,
        "challenger_starts": challenger_starts,
    }


def _wilson_interval(score: float, sample_size: int) -> dict[str, float]:
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


def compute_seat_metrics(entries: list[dict]) -> dict:
    p0_wins = p0_losses = p0_draws = 0
    p1_wins = p1_losses = p1_draws = 0
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
        trajectory_hashes.append(ent["trajectory"])

    p0_total = p0_wins + p0_losses + p0_draws
    p1_total = p1_wins + p1_losses + p1_draws

    traj_counter = Counter(trajectory_hashes)
    duplicate_count = sum(c for c in traj_counter.values() if c > 1)

    p0_score = (p0_wins + 0.5 * p0_draws) / max(p0_total, 1)
    p1_score = (p1_wins + 0.5 * p1_draws) / max(p1_total, 1)
    ds = p0_score - p1_score

    return {
        "p0_score": p0_score,
        "p1_score": p1_score,
        "ds": ds,
        "challenger_starts_0": {
            "games": p0_total,
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "score": p0_score,
            "ci95": _wilson_interval(p0_score, p0_total),
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
            "ci95": _wilson_interval(p1_score, p1_total),
            "margin_mean": statistics.fmean(p1_margins) if p1_margins else 0.0,
            "margin_median": statistics.median(p1_margins) if p1_margins else 0.0,
            "game_length_mean": statistics.fmean(p1_lengths) if p1_lengths else 0.0,
            "game_length_median": statistics.median(p1_lengths) if p1_lengths else 0.0,
        },
        "disadvantaged_seat_score": p1_score,
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
        "duplicate_trajectory_rate": duplicate_count / max(len(trajectory_hashes), 1),
        "total_games": p0_total + p1_total,
    }


def compute_per_opening_metrics(entries: list[dict]) -> list[dict]:
    entries_with_prefix = [
        e for e in entries if e.get("opening_prefix_moves") is not None
    ]
    if not entries_with_prefix:
        return []

    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for e in entries_with_prefix:
        key = ",".join(str(m) for m in e["opening_prefix_moves"])
        by_prefix[key].append(e)

    prefix_metrics: list[dict] = []
    for prefix_key, prefix_entries in by_prefix.items():
        metrics = compute_seat_metrics(prefix_entries)
        prefix_metrics.append(
            {
                "opening_prefix": prefix_key,
                "prefix_length": len(prefix_entries[0]["opening_prefix_moves"]),
                **metrics,
            }
        )

    prefix_metrics.sort(key=lambda m: m["ds"] or 0.0)
    return prefix_metrics


def compute_by_ply_metrics(entries: list[dict]) -> dict[int, dict]:
    entries_with_prefix = [
        e for e in entries if e.get("opening_prefix_moves") is not None
    ]
    by_ply: dict[int, list[dict]] = defaultdict(list)
    for e in entries_with_prefix:
        ply = len(e["opening_prefix_moves"])
        by_ply[ply].append(e)

    result: dict[int, dict] = {}
    for ply, ply_entries in sorted(by_ply.items()):
        result[ply] = compute_seat_metrics(ply_entries)
    return result


BUDGET_PAIR_LABELS = {
    (384, 256): "standard",
    (768, 256): "challenger_768_vs_256",
    (768, 768): "equal_768",
    (1200, 1200): "equal_high",
    (256, 768): "current_high_asymmetry",
}


def candidate_label(candidate_path: str) -> str:
    cp = Path(candidate_path)
    name = cp.name
    for part in cp.parts:
        if part.startswith("replicate_seed_"):
            return part
        if part in (
            "curriculum_ep1",
            "curriculum_ep2",
            "curriculum_lr1e5_ep1",
            "curriculum_lr1e5_ep2",
        ):
            return part
    if "iter0_candidate" in name:
        return "iter0_reference"
    if "control_ep1" in name:
        return "control_ep1"
    if "control_ep2" in name:
        return "control_ep2"
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opening-suite seat-aware benchmark.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--suite", required=True, help="Path to opening suite JSONL.")
    parser.add_argument(
        "--current",
        default="model-artifact/current",
        help="Path to current model artifact.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated candidate artifact paths.",
    )
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,768:768,1200:1200,256:768",
        help="Comma-separated challenger:current simulation budget pairs.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_RUNTIME_C_PUCT,
        help="Global c_puct used when no per-budget override is present.",
    )
    parser.add_argument(
        "--c-puct-schedule-json",
        default=default_runtime_schedule_json(),
        help=(
            "Optional JSON object mapping budget labels like 768:768 to per-budget c_puct overrides. "
            "Default: checked-in runtime schedule. Pass '{}' for no schedule."
        ),
    )
    parser.add_argument(
        "--games-per-opening",
        type=int,
        default=2,
        help="Games per opening prefix (must be ≥2 for seat splits).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--root-policy-mode",
        choices=("deterministic", "visit_count"),
        default="deterministic",
        help="Root move selection policy (deterministic or stochastic via visit_count).",
    )
    parser.add_argument(
        "--root-temperatures",
        default="0.0",
        help="Comma-separated root temperature values (0.0=deterministic, >0.0=stochastic).",
    )
    parser.add_argument(
        "--tactical-root-bias",
        type=float,
        default=None,
        help=(
            "Optional tactical root bias override. Default: unset, which uses the checked-in "
            "arena evaluation default."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds for stochastic evaluations.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per arena invocation (seconds).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    suite = load_suite(args.suite)
    suite_size = len(suite)
    suite_sha = sha256_file(Path(args.suite))
    print(f"Loaded opening suite: {suite_size} openings from {args.suite}")
    print(f"Root policy mode: {args.root_policy_mode}")

    current_sha = sha256_file(Path(args.current) / "weights.json")
    print(f"Current SHA256: {current_sha}")

    candidates = [c.strip() for c in args.candidates.split(",")]
    candidates = [c for c in candidates if c]

    budget_pairs = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c_str, cur_str = bp.split(":", 1)
            budget_pairs.append((int(c_str), int(cur_str)))

    root_temperatures = [float(t.strip()) for t in args.root_temperatures.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    effective_tactical_root_bias = (
        default_eval_tactical_root_bias()
        if args.tactical_root_bias is None
        else float(args.tactical_root_bias)
    )
    cpuct_schedule = parse_cpuct_schedule_json(args.c_puct_schedule_json)
    schedule_manifest = schedule_definition(
        default_c_puct=float(args.c_puct), schedule=cpuct_schedule
    )

    all_temperature_reports: list[dict] = []

    for rt in root_temperatures:
        effective_seeds = [args.seed] if rt <= 0.0 else seeds
        rt_label = f"temp_{rt}".replace(".", "_")
        rt_dir = workdir / rt_label
        rt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Root Temperature: {rt}  Seeds: {effective_seeds}")

        seed_reports: list[dict] = []

        for seed in effective_seeds:
            seed_label = f"seed_{seed}"
            seed_dir = rt_dir / seed_label
            seed_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Seed: {seed}")

            all_candidate_reports: list[dict] = []

            for candidate_path in candidates:
                cand_label = candidate_label(candidate_path)
                cand_dir = seed_dir / cand_label
                cand_dir.mkdir(parents=True, exist_ok=True)

                candidate_sha = sha256_file(Path(candidate_path) / "weights.json")
                print(f"\n  Candidate: {cand_label}  SHA256: {candidate_sha[:12]}...")

                cand_budget_results: dict[str, dict] = {}

                for chall_sims, curr_sims in budget_pairs:
                    effective_c_puct = resolve_budget_cpuct(
                        schedule=cpuct_schedule,
                        challenger_simulations=chall_sims,
                        current_simulations=curr_sims,
                        default_c_puct=float(args.c_puct),
                    )
                    budget_label = BUDGET_PAIR_LABELS.get(
                        (chall_sims, curr_sims), f"{chall_sims}_vs_{curr_sims}"
                    )
                    budget_dir = cand_dir / budget_label
                    budget_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"    Budget {chall_sims}:{curr_sims} ({budget_label}) c_puct={effective_c_puct:.2f} ...",
                        flush=True,
                    )

                    gpo = max(1, args.games_per_opening)
                    total_games = suite_size * gpo
                    arena_seed = seed
                    cache_context = budget_cache_context(
                        suite_path=args.suite,
                        suite_sha=suite_sha,
                        suite_size=suite_size,
                        current_path=args.current,
                        current_sha=current_sha,
                        candidate_path=candidate_path,
                        candidate_sha=candidate_sha,
                        challenger_sims=chall_sims,
                        current_sims=curr_sims,
                        games_per_opening=gpo,
                        total_games=total_games,
                        root_policy_mode=args.root_policy_mode,
                        root_temperature=rt,
                        c_puct=effective_c_puct,
                        c_puct_schedule=schedule_manifest["overrides"],
                        tactical_root_bias=effective_tactical_root_bias,
                        seed=arena_seed,
                    )
                    metrics_path = budget_dir / "metrics.json"
                    if metrics_path.is_file():
                        cached_metrics = load_json(metrics_path)
                        if cache_matches(
                            cached_metrics.get("cache_context", {}), cache_context
                        ):
                            cand_budget_results[budget_label] = cached_metrics
                            ds = cached_metrics["ds"]
                            dup_rate = cached_metrics["duplicate_trajectory_rate"]
                            print(
                                f"      cache hit: DS={ds:+.4f}  P0={cached_metrics['p0_score']:.4f}"
                                f"  P1={cached_metrics['p1_score']:.4f}"
                                f"  dup_traj_rate={dup_rate:.3f}"
                                f"  games={cached_metrics['total_games']}"
                            )
                            continue

                    all_game_entries: list[dict] = []
                    seat_reports: list[dict[str, Any]] = []

                    for seat in (0, 1):
                        seat_label = f"starts_{seat}"
                        seat_dir = budget_dir / seat_label
                        seat_dir.mkdir(parents=True, exist_ok=True)
                        seat_json = str(seat_dir / "arena.json")
                        seat_jsonl = str(seat_dir / "games.jsonl")
                        seat_meta_path = seat_dir / "metadata.json"
                        seat_context = seat_cache_context(
                            cache_context, challenger_starts=seat
                        )

                        suite_jsonl_path = str(seat_dir / "opening_suite.jsonl")
                        with open(suite_jsonl_path, "w", encoding="utf-8") as f:
                            for entry in suite:
                                f.write(
                                    json.dumps({"prefix_moves": entry["prefix_moves"]})
                                    + "\n"
                                )

                        if seat_meta_path.is_file():
                            cached_seat = load_json(seat_meta_path)
                            if (
                                cache_matches(
                                    cached_seat.get("cache_context", {}), seat_context
                                )
                                and Path(seat_json).is_file()
                                and Path(seat_jsonl).is_file()
                            ):
                                game_entries = parse_game_jsonl(seat_jsonl)
                                if game_entries:
                                    all_game_entries.extend(game_entries)
                                    seat_reports.append(load_json(Path(seat_json)))
                                    print(
                                        f"      {seat_label}: cache hit ({len(game_entries)} games)"
                                    )
                                    continue

                        try:
                            t0 = time.time()
                            report = run_arena(
                                challenger=candidate_path,
                                current=args.current,
                                challenger_sims=chall_sims,
                                current_sims=curr_sims,
                                games=total_games,
                                seed=arena_seed,
                                workers=args.workers,
                                out_json=seat_json,
                                out_jsonl=seat_jsonl,
                                opening_prefixes_jsonl=suite_jsonl_path,
                                challenger_starts=seat,
                                games_per_opening=gpo,
                                root_policy_mode=args.root_policy_mode,
                                root_temperature=rt,
                                c_puct=effective_c_puct,
                                tactical_root_bias=effective_tactical_root_bias,
                                timeout=args.timeout,
                            )
                            elapsed = time.time() - t0
                            print(
                                f"      {seat_label}: score={report.get('score', 0):.4f} ({elapsed:.0f}s)"
                            )

                            game_entries = parse_game_jsonl(seat_jsonl)
                            all_game_entries.extend(game_entries)
                            seat_reports.append(report)
                            write_json(
                                seat_meta_path,
                                {
                                    "cache_context": seat_context,
                                    "arena_score": report.get("score"),
                                    "games": len(game_entries),
                                    "elapsed_s": elapsed,
                                },
                            )
                        except RuntimeError as exc:
                            print(f"      {seat_label}: FAILED - {exc}")
                            continue

                    if not all_game_entries:
                        print(f"      No game entries collected for {budget_label}")
                        continue

                    metrics = compute_seat_metrics(all_game_entries)
                    per_opening = compute_per_opening_metrics(all_game_entries)
                    by_ply = compute_by_ply_metrics(all_game_entries)
                    report_notes = {}
                    if seat_reports:
                        report_notes = seat_reports[0].get("notes", {}) or {}
                    budget_result_move_means = [
                        float(report.get("notes", {}).get("move_time_mean_ms"))
                        for report in seat_reports
                        if report.get("notes", {}).get("move_time_mean_ms") is not None
                    ]
                    budget_result_move_p95s = [
                        float(report.get("notes", {}).get("move_time_p95_ms"))
                        for report in seat_reports
                        if report.get("notes", {}).get("move_time_p95_ms") is not None
                    ]

                    budget_result = {
                        **metrics,
                        "cache_context": cache_context,
                        "per_opening_metrics": per_opening,
                        "by_ply_metrics": {str(k): v for k, v in by_ply.items()},
                        "budget_label": budget_label,
                        "challenger_simulations": chall_sims,
                        "current_simulations": curr_sims,
                        "effective_c_puct": effective_c_puct,
                        "tactical_root_bias": effective_tactical_root_bias,
                        "search_profile": report_notes.get("search_profile"),
                        "search_profile_hash": report_notes.get("search_profile_hash"),
                        "move_time_mean_ms": statistics.fmean(budget_result_move_means)
                        if budget_result_move_means
                        else None,
                        "move_time_p95_ms": statistics.fmean(budget_result_move_p95s)
                        if budget_result_move_p95s
                        else None,
                        "total_games": len(all_game_entries),
                    }

                    if per_opening:
                        worst_10 = per_opening[:10]
                        best_10 = per_opening[-10:]
                        budget_result["worst_10_openings"] = worst_10
                        budget_result["best_10_openings"] = best_10

                    cand_budget_results[budget_label] = budget_result

                    write_json(metrics_path, budget_result)

                    ds = metrics["ds"]
                    dup_rate = metrics["duplicate_trajectory_rate"]
                    print(
                        f"      DS={ds:+.4f}  P0={metrics['p0_score']:.4f}  P1={metrics['p1_score']:.4f}"
                        f"  dup_traj_rate={dup_rate:.3f}"
                        f"  games={metrics['total_games']}"
                    )

                candidate_report = {
                    "candidate": cand_label,
                    "candidate_path": candidate_path,
                    "candidate_sha256": candidate_sha,
                    "current_sha256": current_sha,
                    "suite_size": suite_size,
                    "budget_results": cand_budget_results,
                }
                all_candidate_reports.append(candidate_report)

            ranking_rows = _build_ranking_table(all_candidate_reports)
            print(f"\n  Seed {seed} -- Candidate Ranking")
            for row in ranking_rows:
                std_p0 = row.get("std_p0")
                std_p1 = row.get("std_p1")
                std_ds = row.get("std_ds")
                eqhi_ds = row.get("equal_high_ds")
                print(
                    f"    {row['candidate']:<30} P0={std_p0:.4f} "
                    if std_p0 is not None
                    else f"    {row['candidate']:<30} P0=N/A P1={std_p1:.4f} "
                    if std_p1 is not None
                    else f"P1=N/A DS={std_ds:+.4f} "
                    if std_ds is not None
                    else f"DS=N/A eqhi={eqhi_ds:+.4f}"
                    if eqhi_ds is not None
                    else "eqhi=N/A"
                )

            seed_report = {
                "root_temperature": rt,
                "seed": seed,
                "candidate_reports": all_candidate_reports,
                "ranking": ranking_rows,
            }
            seed_reports.append(seed_report)

        seed_summary = None
        if len(effective_seeds) > 1:
            seed_summary = _summarize_across_seeds(seed_reports)
            summary_path = rt_dir / "seed_summary.json"
            summary_path.write_text(
                json.dumps(seed_summary, indent=2), encoding="utf-8"
            )
            print(f"\n  Seed summary written to {summary_path}")

        temperature_report = {
            "root_temperature": rt,
            "seed_reports": seed_reports,
        }
        if seed_summary is not None:
            temperature_report["seed_summary"] = seed_summary
        all_temperature_reports.append(temperature_report)

    full_report_path = workdir / "temperature_benchmark_report.json"
    write_json(
        full_report_path,
        {
            "suite_path": args.suite,
            "suite_sha256": suite_sha,
            "suite_size": suite_size,
            "current_sha256": current_sha,
            "c_puct": float(args.c_puct),
            "c_puct_schedule": schedule_manifest,
            "root_policy_mode": args.root_policy_mode,
            "temperature_reports": all_temperature_reports,
        },
    )
    print(f"\nFull temperature benchmark report written to {full_report_path}")
    return 0


def _summarize_across_seeds(seed_reports: list[dict]) -> dict:
    candidate_names = sorted(
        {r["candidate"] for sr in seed_reports for r in sr.get("candidate_reports", [])}
    )
    budget_labels = sorted(
        {
            b
            for sr in seed_reports
            for cr in sr.get("candidate_reports", [])
            for b in cr.get("budget_results", {})
        }
    )
    summaries: list[dict] = []
    for cand in candidate_names:
        entry: dict = {"candidate": cand}
        for bl in budget_labels:
            ds_vals = []
            p0_vals = []
            p1_vals = []
            dup_vals = []
            for sr in seed_reports:
                for cr in sr.get("candidate_reports", []):
                    if cr.get("candidate") != cand:
                        continue
                    br = cr.get("budget_results", {}).get(bl)
                    if br:
                        ds_vals.append(br.get("ds", 0.0))
                        p0_vals.append(br.get("p0_score", 0.5))
                        p1_vals.append(br.get("p1_score", 0.5))
                        dup_vals.append(br.get("duplicate_trajectory_rate", 0.0))
            if ds_vals:
                entry[f"{bl}_mean_ds"] = statistics.mean(ds_vals)
                entry[f"{bl}_std_ds"] = (
                    statistics.stdev(ds_vals) if len(ds_vals) > 1 else 0.0
                )
                entry[f"{bl}_mean_p0"] = statistics.mean(p0_vals)
                entry[f"{bl}_mean_p1"] = statistics.mean(p1_vals)
                if dup_vals:
                    entry[f"{bl}_mean_dup_rate"] = statistics.mean(dup_vals)
        summaries.append(entry)
    return {"candidate_summaries": summaries, "num_seeds": len(seed_reports)}


def _build_ranking_table(candidate_reports: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for report in candidate_reports:
        budget_results = report.get("budget_results", {})
        std = budget_results.get("standard", {})
        eq_hi = budget_results.get("equal_high", {})
        eq_768 = budget_results.get("equal_768", {})
        curr_hi = budget_results.get("current_high_asymmetry", {})
        ch_768 = budget_results.get("challenger_768_vs_256", {})

        row = {
            "candidate": report.get("candidate", "unknown"),
            "candidate_sha256": report.get("candidate_sha256", ""),
            "std_p0": std.get("p0_score"),
            "std_p1": std.get("p1_score"),
            "std_ds": std.get("ds"),
            "std_disadvantaged": std.get("disadvantaged_seat_score"),
            "equal_high_p0": eq_hi.get("p0_score"),
            "equal_high_p1": eq_hi.get("p1_score"),
            "equal_high_ds": eq_hi.get("ds"),
            "equal_high_disadvantaged": eq_hi.get("disadvantaged_seat_score"),
            "equal_768_ds": eq_768.get("ds"),
            "current_high_ds": curr_hi.get("ds"),
            "challenger_768_ds": ch_768.get("ds"),
            "std_dup_traj_rate": std.get("duplicate_trajectory_rate"),
        }
        rows.append(row)

    def rank_key(row: dict) -> tuple:
        std_ds = float(row.get("std_ds") or 0.0)
        eq_ds = float(row.get("equal_high_ds") or 0.0)
        return (-std_ds, -eq_ds)

    return sorted(rows, key=rank_key, reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
