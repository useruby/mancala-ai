#!/usr/bin/env python3
"""Opening-randomized seat-aware diagnostic for AlphaZero-lite.

Evaluates a candidate vs current across multiple random opening prefixes,
search budgets, and opening ply counts to determine whether high-search
disadvantaged-seat breakthrough generalizes beyond the deterministic
opening.

Does not train, promote, or overwrite any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _find_python() -> str:
    candidates = [
        REPO_ROOT / ".venv/bin/python",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def compute_per_prefix_metrics(entries: list[dict]) -> list[dict]:
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
        metrics = compute_seat_split_metrics(prefix_entries)
        prefix_metrics.append(
            {
                "opening_prefix": prefix_key,
                "prefix_length": len(prefix_entries[0]["opening_prefix_moves"]),
                **metrics,
            }
        )

    prefix_metrics.sort(
        key=lambda m: m["disadvantaged_seat_score"] or 0.0,
    )
    return prefix_metrics


def compute_first_move_distribution(entries: list[dict]) -> dict:
    first_moves: Counter = Counter()
    for e in entries:
        fm = e.get("first_move_challenger")
        if fm is not None:
            first_moves[str(fm)] += 1
        fm = e.get("first_move_current")
        if fm is not None:
            first_moves[str(fm)] += 1
    return dict(first_moves.most_common(12))


def run_arena(
    challenger: str,
    current: str,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    out_json: str,
    out_jsonl: str,
    *,
    random_opening_plies: int = 0,
    opening_seed: int | None = None,
    opening_samples: int = 0,
    games_per_opening: int = 2,
    challenger_starts: int | None = None,
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
        "--random-opening-plies",
        str(random_opening_plies),
    ]
    if opening_seed is not None:
        cmd.extend(["--opening-seed", str(opening_seed)])
    if opening_samples > 0:
        cmd.extend(["--opening-samples", str(opening_samples)])
    if games_per_opening != 2:
        cmd.extend(["--games-per-opening", str(games_per_opening)])
    if challenger_starts is not None:
        cmd.extend(["--challenger-starts", str(challenger_starts)])

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Opening-randomized seat-aware diagnostic."
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Working directory for arena outputs.",
    )
    parser.add_argument(
        "--current",
        default="model-artifact/current",
        help="Path to current model artifact.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate model artifact.",
    )
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,768:768,1200:1200,256:768",
        help="Comma-separated challenger:current simulation budget pairs.",
    )
    parser.add_argument(
        "--random-opening-plies",
        default="0,2,4,6",
        help="Comma-separated ply counts for random openings.",
    )
    parser.add_argument(
        "--opening-samples",
        type=int,
        default=64,
        help="Number of distinct random opening prefixes.",
    )
    parser.add_argument(
        "--opening-seed",
        type=int,
        default=47,
        help="Seed for random opening generation.",
    )
    parser.add_argument(
        "--games-per-opening",
        type=int,
        default=2,
        help="Games per opening prefix.",
    )
    parser.add_argument(
        "--forced-seat-splits",
        action="store_true",
        help="Run separate invocations for forced seat 0 and 1.",
    )
    parser.add_argument(
        "--seed", type=int, default=47, help="Base random seed for arena."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    candidate_sha = sha256_file(Path(args.candidate) / "weights.json")
    current_sha = sha256_file(Path(args.current) / "weights.json")

    ply_counts = [int(x.strip()) for x in args.random_opening_plies.split(",")]
    budget_pairs = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c_str, cur_str = bp.split(":", 1)
            budget_pairs.append((int(c_str), int(cur_str)))

    BUDGET_LABELS = {
        (384, 256): "standard",
        (768, 256): "challenger_768_vs_256",
        (768, 768): "equal_768",
        (1200, 1200): "equal_high",
        (256, 768): "current_high_asymmetry",
    }

    all_results: list[dict] = []
    for ply_count in ply_counts:
        print(f"\n[opening-diag] ply_count={ply_count} ...", flush=True)

        for chall_sims, curr_sims in budget_pairs:
            budget_label = BUDGET_LABELS.get(
                (chall_sims, curr_sims), f"{chall_sims}_vs_{curr_sims}"
            )
            pair_dir = workdir / f"ply{ply_count}" / budget_label
            pair_dir.mkdir(parents=True, exist_ok=True)

            opening_seed_val = args.opening_seed + ply_count * 1000
            if ply_count == 0:
                total_games = 2
                effective_opening_samples = 0
            else:
                total_games = args.opening_samples * args.games_per_opening
                effective_opening_samples = args.opening_samples

            if args.forced_seat_splits:
                for forced_seat in (0, 1):
                    ss_label = f"starts_{forced_seat}"
                    ss_dir = pair_dir / ss_label
                    ss_dir.mkdir(parents=True, exist_ok=True)
                    ss_json = str(ss_dir / "arena.json")
                    ss_jsonl = str(ss_dir / "games.jsonl")

                    print(
                        f"  [opening-diag] {budget_label} forced_seat={forced_seat} ...",
                        flush=True,
                    )
                    try:
                        report = run_arena(
                            challenger=str(args.candidate),
                            current=args.current,
                            challenger_sims=chall_sims,
                            current_sims=curr_sims,
                            games=total_games,
                            seed=args.seed + forced_seat,
                            workers=args.workers,
                            out_json=ss_json,
                            out_jsonl=ss_jsonl,
                            random_opening_plies=ply_count,
                            opening_seed=opening_seed_val,
                            opening_samples=effective_opening_samples,
                            games_per_opening=args.games_per_opening,
                            challenger_starts=forced_seat,
                        )
                    except RuntimeError as e:
                        print(f"    FAILED: {e}", flush=True)
                        all_results.append(
                            {
                                "ply_count": ply_count,
                                "budget_label": budget_label,
                                "forced_seat": forced_seat,
                                "error": str(e),
                            }
                        )
                        continue

                    fs_entries = parse_game_jsonl(ss_jsonl)
                    fs_seat_metrics = compute_seat_split_metrics(fs_entries)
                    fs_prefix_metrics = compute_per_prefix_metrics(fs_entries)
                    fs_first_move_dist = compute_first_move_distribution(fs_entries)

                    all_results.append(
                        {
                            "ply_count": ply_count,
                            "budget_label": budget_label,
                            "challenger_sims": chall_sims,
                            "current_sims": curr_sims,
                            "forced_seat": forced_seat,
                            "arena_score": report["score"],
                            "arena_wins": report["wins"],
                            "arena_losses": report["losses"],
                            "arena_draws": report["draws"],
                            "seat_metrics": fs_seat_metrics,
                            "prefix_metrics": fs_prefix_metrics,
                            "first_move_dist": fs_first_move_dist,
                            "move_time_mean_ms": (
                                report.get("notes", {}).get("move_time_mean_ms")
                                if isinstance(report.get("notes"), dict)
                                else None
                            ),
                            "move_time_p95_ms": (
                                report.get("notes", {}).get("move_time_p95_ms")
                                if isinstance(report.get("notes"), dict)
                                else None
                            ),
                        }
                    )
            else:
                alt_json = str(pair_dir / "alternating_arena.json")
                alt_jsonl = str(pair_dir / "alternating_games.jsonl")

                print(f"  [opening-diag] {budget_label} alternating ...", flush=True)
                try:
                    report = run_arena(
                        challenger=str(args.candidate),
                        current=args.current,
                        challenger_sims=chall_sims,
                        current_sims=curr_sims,
                        games=total_games,
                        seed=args.seed,
                        workers=args.workers,
                        out_json=alt_json,
                        out_jsonl=alt_jsonl,
                        random_opening_plies=ply_count,
                        opening_seed=opening_seed_val,
                        opening_samples=effective_opening_samples,
                        games_per_opening=args.games_per_opening,
                    )
                except RuntimeError as e:
                    print(f"    FAILED: {e}", flush=True)
                    all_results.append(
                        {
                            "ply_count": ply_count,
                            "budget_label": budget_label,
                            "error": str(e),
                        }
                    )
                    continue

                alt_entries = parse_game_jsonl(alt_jsonl)
                seat_metrics = compute_seat_split_metrics(alt_entries)
                prefix_metrics = compute_per_prefix_metrics(alt_entries)
                first_move_dist = compute_first_move_distribution(alt_entries)

                all_results.append(
                    {
                        "ply_count": ply_count,
                        "budget_label": budget_label,
                        "challenger_sims": chall_sims,
                        "current_sims": curr_sims,
                        "arena_score": report["score"],
                        "arena_wins": report["wins"],
                        "arena_losses": report["losses"],
                        "arena_draws": report["draws"],
                        "seat_metrics": seat_metrics,
                        "prefix_metrics": prefix_metrics,
                        "first_move_dist": first_move_dist,
                        "move_time_mean_ms": (
                            report.get("notes", {}).get("move_time_mean_ms")
                            if isinstance(report.get("notes"), dict)
                            else None
                        ),
                        "move_time_p95_ms": (
                            report.get("notes", {}).get("move_time_p95_ms")
                            if isinstance(report.get("notes"), dict)
                            else None
                        ),
                    }
                )

    ds_by_ply_and_budget: list[dict] = []
    for result in all_results:
        if "error" in result:
            continue
        sm = result["seat_metrics"]
        prefix_entries = result.get("prefix_metrics", [])
        ds_values = [
            m["disadvantaged_seat_score"]
            for m in prefix_entries
            if m.get("disadvantaged_seat_score") is not None
        ]
        ds_variance = statistics.variance(ds_values) if len(ds_values) >= 2 else 0.0

        forced_seat = result.get("forced_seat")

        ds_by_ply_and_budget.append(
            {
                "ply_count": result["ply_count"],
                "budget_label": result["budget_label"],
                "challenger_sims": result.get("challenger_sims"),
                "current_sims": result.get("current_sims"),
                "forced_seat": forced_seat,
                "disadvantaged_seat_score": sm["disadvantaged_seat_score"],
                "challenger_p0_score": sm["challenger_starts_0"]["score"],
                "challenger_p1_score": sm["challenger_starts_1"]["score"],
                "p0_wins": sm["challenger_starts_0"]["wins"],
                "p0_losses": sm["challenger_starts_0"]["losses"],
                "p0_draws": sm["challenger_starts_0"]["draws"],
                "p1_wins": sm["challenger_starts_1"]["wins"],
                "p1_losses": sm["challenger_starts_1"]["losses"],
                "p1_draws": sm["challenger_starts_1"]["draws"],
                "margin_mean": sm["margin_mean"],
                "margin_median": sm["margin_median"],
                "game_length_mean": sm["game_length_mean"],
                "game_length_median": sm["game_length_median"],
                "unique_trajectories": sm["unique_trajectories"],
                "duplicate_trajectory_count": sm["duplicate_trajectory_count"],
                "total_games": sm["total_games"],
                "prefix_ds_variance": ds_variance,
                "num_prefixes": len(prefix_entries),
                "first_move_dist": result.get("first_move_dist", {}),
                "move_time_p95_ms": result.get("move_time_p95_ms"),
            }
        )

    ds_by_ply_and_budget.sort(
        key=lambda r: (
            r["ply_count"],
            r.get("budget_label", ""),
            r.get("forced_seat") or -1,
        )
    )

    worst_prefixes: list[dict] = []
    best_prefixes: list[dict] = []
    for result in all_results:
        if "error" in result:
            continue
        for pm in result.get("prefix_metrics", []):
            entry = {
                "ply_count": result["ply_count"],
                "budget_label": result["budget_label"],
                "forced_seat": result.get("forced_seat"),
                "opening_prefix": pm["opening_prefix"],
                "prefix_length": pm["prefix_length"],
                "disadvantaged_seat_score": pm["disadvantaged_seat_score"],
                "challenger_p0_score": pm["challenger_starts_0"]["score"],
                "challenger_p1_score": pm["challenger_starts_1"]["score"],
                "p0_wins": pm["challenger_starts_0"]["wins"],
                "p1_wins": pm["challenger_starts_1"]["wins"],
                "total_games": pm["total_games"],
            }
            worst_prefixes.append(entry)
            best_prefixes.append(entry)

    worst_prefixes.sort(
        key=lambda m: m["disadvantaged_seat_score"] or 0.0,
    )
    best_prefixes.sort(
        key=lambda m: m["disadvantaged_seat_score"] or 0.0,
        reverse=True,
    )

    worst_10 = worst_prefixes[:10]
    best_10 = best_prefixes[:10]

    ds_1200_1200_by_ply: dict[int, float] = {}
    for row in ds_by_ply_and_budget:
        if (
            row.get("challenger_sims") == 1200
            and row.get("current_sims") == 1200
            and row.get("forced_seat") is None
        ):
            ds_1200_1200_by_ply[row["ply_count"]] = row["disadvantaged_seat_score"]

    has_robust_1200 = all(
        ds_1200_1200_by_ply.get(p, 0.0) >= 0.9
        for p in [ply for ply in ply_counts if ply > 0]
    )
    has_any_high_ds_at_768_256 = any(
        row.get("challenger_sims") == 768
        and row.get("current_sims") == 256
        and row.get("forced_seat") is None
        and row["disadvantaged_seat_score"] > 0.0
        for row in ds_by_ply_and_budget
        if row["ply_count"] > 0
    )
    has_collapse = any(
        row["ply_count"] > 0
        and row.get("challenger_sims") == 1200
        and row.get("current_sims") == 1200
        and row.get("forced_seat") is None
        and row["disadvantaged_seat_score"] < 0.5
        for row in ds_by_ply_and_budget
    )
    if has_robust_1200:
        classification = "opening_robust_high_search"
    elif has_collapse:
        classification = "deterministic_opening_artifact"
    elif has_any_high_ds_at_768_256:
        classification = "opening_curriculum_promising"
    else:
        classification = (
            "opening_curriculum_promising"
            if has_any_high_ds_at_768_256
            else "inconclusive"
        )

    illegal_prefix_count = 0
    for result in all_results:
        if "error" in result:
            continue
        pm = result.get("prefix_metrics", [])
        for m in pm:
            if (
                m.get("prefix_length", 0) != result["ply_count"]
                and result["ply_count"] > 0
            ):
                illegal_prefix_count += 1

    report = {
        "schema": "azlite_opening_randomized_seat_diagnostic_v1",
        "candidate_path": args.candidate,
        "candidate_sha256": candidate_sha,
        "current_path": args.current,
        "current_sha256": current_sha,
        "classification": classification,
        "opening_ply_counts": ply_counts,
        "opening_samples": args.opening_samples,
        "opening_seed": args.opening_seed,
        "games_per_opening": args.games_per_opening,
        "forced_seat_splits": args.forced_seat_splits,
        "illegal_or_skipped_opening_count": illegal_prefix_count,
        "num_unique_opening_prefixes": args.opening_samples,
        "ds_by_ply_and_budget": ds_by_ply_and_budget,
        "ds_at_1200_1200_by_ply": ds_1200_1200_by_ply,
        "worst_10_opening_prefixes": worst_10,
        "best_10_opening_prefixes": best_10,
    }

    print(f"\nClassification: {classification}")
    print(f"Candidate SHA256: {candidate_sha}")
    print(f"Current SHA256: {current_sha}")
    print(f"Illegal/skipped openings: {illegal_prefix_count}")
    print("\nDS by ply count and budget:")
    for row in ds_by_ply_and_budget:
        fs = (
            f" forced={row['forced_seat']}"
            if row.get("forced_seat") is not None
            else ""
        )
        print(
            f"  ply={row['ply_count']} {row['budget_label']}{fs}: "
            f"DS={row['disadvantaged_seat_score']:.4f} "
            f"(P0={row['challenger_p0_score']:.4f} P1={row['challenger_p1_score']:.4f})"
        )
    print(f"\nDS at 1200:1200 by ply: {ds_1200_1200_by_ply}")

    if worst_10:
        print("\nWorst 10 prefixes (lowest DS):")
        for wp in worst_10[:5]:
            print(
                f"  ply={wp['ply_count']} {wp['budget_label']} ds={wp['disadvantaged_seat_score']:.4f} "
                f"prefix={wp['opening_prefix'][:40]}..."
            )
    if best_10:
        print("\nBest 10 prefixes (highest DS):")
        for bp2 in best_10[:5]:
            print(
                f"  ply={bp2['ply_count']} {bp2['budget_label']} ds={bp2['disadvantaged_seat_score']:.4f} "
                f"prefix={bp2['opening_prefix'][:40]}..."
            )

    out_path = workdir / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
