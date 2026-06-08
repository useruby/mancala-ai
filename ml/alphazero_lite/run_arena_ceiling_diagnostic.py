#!/usr/bin/env python3
"""Diagnose the 0.50 arena ceiling with search-budget and seat-split analysis.

Preserves existing arena.py behavior and local_promotion_gate behavior.
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
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arena ceiling diagnostic: search-budget matrix and seat-split analysis."
    )
    parser.add_argument(
        "--workdir", required=True, help="Working directory for output artifacts"
    )
    parser.add_argument(
        "--current", required=True, help="Path to current production artifact"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated paths to candidate artifacts",
    )
    parser.add_argument(
        "--budget-pairs",
        default="128:128,256:256,384:384,768:768,1200:1200,384:256,768:256,1200:256,256:768",
        help="Comma-separated challenger:current simulation pairs",
    )
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--extended-games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-arenas",
        action="store_true",
        help="Skip arena runs (use existing JSONL)",
    )
    parser.add_argument(
        "--skip-raw-policy", action="store_true", help="Skip raw-policy diagnostic"
    )
    parser.add_argument(
        "--source-states-files",
        default="/tmp/azlite_random_teacher_quality/random_source_states.jsonl,/tmp/azlite_iterative_random_replay/iter1_candidate_source_states.jsonl",
        help="Comma-separated paths to source state files for raw-policy diagnostic",
    )
    parser.add_argument("--raw-policy-samples", type=int, default=500)
    parser.add_argument(
        "--out-report", default=None, help="Override output report path"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval_95(score: float, sample_size: int) -> dict[str, float]:
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


def candidate_name(path: str) -> str:
    p = Path(path)
    if "iter0_candidate" in str(p) or "iter0_candidate_artifact" in str(p):
        return "iter0_reference"
    if "iter1_continue_no_new_data" in str(p):
        return "iter1_continue_no_new_data"
    if "iter1_candidate_random_replay" in str(p):
        return "iter1_candidate_random_replay"
    return p.name


def run_arena(
    challenger: str,
    current: str,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    out_path: str,
    game_jsonl_path: str,
    challenger_starts: int | None = None,
) -> dict:
    workers_arg = max(1, workers)
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
        str(workers_arg),
        "--min-score",
        "0.0",
        "--out",
        out_path,
        "--game-jsonl",
        game_jsonl_path,
    ]
    if challenger_starts is not None:
        cmd.extend(["--challenger-starts", str(challenger_starts)])

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=max(600, games * 60),
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"arena failed: {stderr or ' '.join(cmd)}")
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


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
    margins: list[int] = []
    game_lengths: list[int] = []
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
        else:
            if winner == "challenger":
                p1_wins += 1
            elif winner == "current":
                p1_losses += 1
            else:
                p1_draws += 1

        if ent.get("first_move_challenger") is not None:
            first_moves_challenger[str(ent["first_move_challenger"])] += 1
        if ent.get("first_move_current") is not None:
            first_moves_current[str(ent["first_move_current"])] += 1
        margins.append(margin)
        game_lengths.append(ent["game_length"])
        trajectory_hashes.append(ent["trajectory"])

    p0_total = p0_wins + p0_losses + p0_draws
    p1_total = p1_wins + p1_losses + p1_draws

    traj_counter = Counter(trajectory_hashes)
    duplicate_count = sum(c for c in traj_counter.values() if c > 1)

    return {
        "challenger_starts_0": {
            "games": p0_total,
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "score": (p0_wins + 0.5 * p0_draws) / max(p0_total, 1),
        },
        "challenger_starts_1": {
            "games": p1_total,
            "wins": p1_wins,
            "losses": p1_losses,
            "draws": p1_draws,
            "score": (p1_wins + 0.5 * p1_draws) / max(p1_total, 1),
        },
        "first_move_challenger_dist": dict(first_moves_challenger.most_common(6)),
        "first_move_current_dist": dict(first_moves_current.most_common(6)),
        "margin_mean": statistics.fmean(margins) if margins else 0.0,
        "margin_median": statistics.median(margins) if margins else 0.0,
        "game_length_mean": statistics.fmean(game_lengths) if game_lengths else 0.0,
        "game_length_median": statistics.median(game_lengths) if game_lengths else 0.0,
        "unique_trajectories": len(traj_counter),
        "duplicate_trajectory_count": duplicate_count,
    }


def run_all_arenas(args: argparse.Namespace) -> list[dict]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    candidate_paths = [p.strip() for p in args.candidates.split(",") if p.strip()]
    budget_pairs = []
    for bp in args.budget_pairs.split(","):
        bp = bp.strip()
        if ":" in bp:
            c, cur = bp.split(":", 1)
            budget_pairs.append((int(c), int(cur)))

    results = []
    for cand_path in candidate_paths:
        name = candidate_name(cand_path)
        cand_sha = sha256_file(Path(cand_path) / "weights.json")
        current_sha = sha256_file(Path(args.current) / "weights.json")

        for chall_sims, curr_sims in budget_pairs:
            pair_label = f"{chall_sims}_vs_{curr_sims}"
            pair_dir = workdir / name / pair_label
            pair_dir.mkdir(parents=True, exist_ok=True)

            out_path = str(pair_dir / "arena.json")
            jsonl_path = str(pair_dir / "games.jsonl")

            if not args.skip_arenas:
                print(f"[arena] {name} {pair_label} ...", flush=True)
                try:
                    arena_report = run_arena(
                        challenger=cand_path,
                        current=args.current,
                        challenger_sims=chall_sims,
                        current_sims=curr_sims,
                        games=args.games,
                        seed=args.seed,
                        workers=args.workers,
                        out_path=out_path,
                        game_jsonl_path=jsonl_path,
                    )
                except RuntimeError as e:
                    print(f"  FAILED: {e}", flush=True)
                    results.append(
                        {
                            "candidate": name,
                            "candidate_path": cand_path,
                            "candidate_sha256": cand_sha,
                            "current_sha256": current_sha,
                            "challenger_simulations": chall_sims,
                            "current_simulations": curr_sims,
                            "error": str(e),
                        }
                    )
                    continue
            else:
                arena_report = json.loads(Path(out_path).read_text(encoding="utf-8"))
                print(f"[arena] {name} {pair_label} (cached)", flush=True)

            entries = parse_game_jsonl(jsonl_path)
            seat_metrics = compute_seat_split_metrics(entries)

            arena_notes = arena_report.get("notes", {})
            row = {
                "candidate": name,
                "candidate_path": cand_path,
                "candidate_sha256": cand_sha,
                "current_sha256": current_sha,
                "challenger_simulations": chall_sims,
                "current_simulations": curr_sims,
                "games": arena_report["games_played"],
                "wins": arena_report["wins"],
                "losses": arena_report["losses"],
                "draws": arena_report["draws"],
                "score": arena_report["score"],
                "ci95": wilson_interval_95(
                    arena_report["score"], arena_report["games_played"]
                ),
                **seat_metrics,
                "move_time_mean_ms": arena_notes.get("move_time_mean_ms"),
                "move_time_p95_ms": arena_notes.get("move_time_p95_ms"),
            }
            results.append(row)

            score = arena_report["score"]
            in_middle = 0.45 <= score <= 0.55
            if in_middle:
                ext_dir = workdir / name / f"{pair_label}_extended"
                ext_dir.mkdir(parents=True, exist_ok=True)
                ext_out = str(ext_dir / "arena.json")
                ext_jsonl = str(ext_dir / "games.jsonl")
                if not args.skip_arenas:
                    print(
                        f"[arena] {name} {pair_label} EXTENDED {args.extended_games}g ...",
                        flush=True,
                    )
                    try:
                        ext_report = run_arena(
                            challenger=cand_path,
                            current=args.current,
                            challenger_sims=chall_sims,
                            current_sims=curr_sims,
                            games=args.extended_games,
                            seed=args.seed + 1000,
                            workers=args.workers,
                            out_path=ext_out,
                            game_jsonl_path=ext_jsonl,
                        )
                    except RuntimeError as e:
                        print(f"  EXTENDED FAILED: {e}", flush=True)
                        continue
                else:
                    ext_report = json.loads(Path(ext_out).read_text(encoding="utf-8"))
                ext_entries = parse_game_jsonl(ext_jsonl)
                ext_seat = compute_seat_split_metrics(ext_entries)
                ext_notes = ext_report.get("notes", {})
                row["extended"] = {
                    "games": ext_report["games_played"],
                    "wins": ext_report["wins"],
                    "losses": ext_report["losses"],
                    "draws": ext_report["draws"],
                    "score": ext_report["score"],
                    "ci95": wilson_interval_95(
                        ext_report["score"], ext_report["games_played"]
                    ),
                    **{f"extended_{k}": v for k, v in ext_seat.items()},
                    "move_time_mean_ms": ext_notes.get("move_time_mean_ms"),
                    "move_time_p95_ms": ext_notes.get("move_time_p95_ms"),
                }

            # Run forced seat-split: challenger starts 0, then starts 1
            for forced_starts in (0, 1):
                ss_dir = workdir / name / f"{pair_label}_starts_{forced_starts}"
                ss_dir.mkdir(parents=True, exist_ok=True)
                ss_out = str(ss_dir / "arena.json")
                ss_jsonl = str(ss_dir / "games.jsonl")
                ss_games = args.games
                if not args.skip_arenas:
                    print(
                        f"[arena] {name} {pair_label} starts={forced_starts} ...",
                        flush=True,
                    )
                    try:
                        run_arena(
                            challenger=cand_path,
                            current=args.current,
                            challenger_sims=chall_sims,
                            current_sims=curr_sims,
                            games=ss_games,
                            seed=args.seed + 2000 + forced_starts,
                            workers=args.workers,
                            out_path=ss_out,
                            game_jsonl_path=ss_jsonl,
                            challenger_starts=forced_starts,
                        )
                    except RuntimeError as e:
                        print(f"  SEAT-SPLIT FAILED: {e}", flush=True)
                        continue
                ss_report = json.loads(Path(ss_out).read_text(encoding="utf-8"))
                key = f"forced_starts_{forced_starts}"
                row[key] = {
                    "games": ss_report["games_played"],
                    "wins": ss_report["wins"],
                    "losses": ss_report["losses"],
                    "draws": ss_report["draws"],
                    "score": ss_report["score"],
                }

    return results


def run_raw_policy_diagnostic(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from ml.alphazero_lite.arena import ArtifactEvaluator
    from ml.alphazero_lite.kalah_rules import KalahGame

    source_files = [p.strip() for p in args.source_states_files.split(",") if p.strip()]
    candidate_paths = [p.strip() for p in args.candidates.split(",") if p.strip()]

    all_states: list[dict] = []
    for sf in source_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_states.append(json.loads(line))

    import random

    rng = random.Random(args.seed)
    sample_size = min(args.raw_policy_samples, len(all_states))
    sampled = rng.sample(all_states, sample_size)

    current_eval = ArtifactEvaluator(Path(args.current))
    candidate_evals: dict[str, ArtifactEvaluator] = {}
    for cp in candidate_paths:
        name = candidate_name(cp)
        candidate_evals[name] = ArtifactEvaluator(Path(cp))

    results: dict[str, dict] = {}
    for cand_name, cand_eval in candidate_evals.items():
        agreements: list[bool] = []
        cand_teacher_agreements: list[bool] = []
        curr_teacher_agreements: list[bool] = []
        cand_entropies: list[float] = []
        curr_entropies: list[float] = []
        cand_values: list[float] = []
        curr_values: list[float] = []

        for row in sampled:
            game_state = row.get("game_state")
            if not isinstance(game_state, dict):
                continue
            game = KalahGame.from_state(game_state)
            cand_priors, cand_value = cand_eval.evaluate(game)
            curr_priors, curr_value = current_eval.evaluate(game)

            cand_top = int(cand_priors.argmax())
            curr_top = int(curr_priors.argmax())
            teacher_top = row.get("teacher_top_move")

            agreements.append(cand_top == curr_top)
            cand_values.append(cand_value)
            curr_values.append(curr_value)

            nonzero = cand_priors[cand_priors > 0]
            if len(nonzero) > 0:
                cand_entropy = -float(np.sum(nonzero * np.log(nonzero + 1e-12)))
            else:
                cand_entropy = 0.0
            nonzero_curr = curr_priors[curr_priors > 0]
            if len(nonzero_curr) > 0:
                curr_entropy = -float(
                    np.sum(nonzero_curr * np.log(nonzero_curr + 1e-12))
                )
            else:
                curr_entropy = 0.0
            cand_entropies.append(cand_entropy)
            curr_entropies.append(curr_entropy)

            if teacher_top is not None:
                cand_teacher_agreements.append(cand_top == int(teacher_top))
                curr_teacher_agreements.append(curr_top == int(teacher_top))

        results[cand_name] = {
            "sampled_states": len(cand_entropies),
            "candidate_current_top_move_agreement": statistics.fmean(agreements)
            if agreements
            else None,
            "candidate_classic_mcts_top_move_agreement": (
                statistics.fmean(cand_teacher_agreements)
                if cand_teacher_agreements
                else None
            ),
            "current_classic_mcts_top_move_agreement": (
                statistics.fmean(curr_teacher_agreements)
                if curr_teacher_agreements
                else None
            ),
            "candidate_policy_entropy_mean": statistics.fmean(cand_entropies)
            if cand_entropies
            else None,
            "current_policy_entropy_mean": statistics.fmean(curr_entropies)
            if curr_entropies
            else None,
            "candidate_value_mean": statistics.fmean(cand_values)
            if cand_values
            else None,
            "candidate_value_std": statistics.pstdev(cand_values)
            if len(cand_values) > 1
            else None,
            "current_value_mean": statistics.fmean(curr_values)
            if curr_values
            else None,
            "current_value_std": statistics.pstdev(curr_values)
            if len(curr_values) > 1
            else None,
        }

    return results


def classify_results(results: list[dict], raw_policy: dict) -> str:
    classifications = set()

    for row in results:
        if "error" in row:
            continue
        seat0 = row.get("challenger_starts_0", {})
        seat1 = row.get("challenger_starts_1", {})
        dup_traj = row.get("duplicate_trajectory_count", 0)
        games = row.get("games", 1)

        if seat0.get("score", 0.5) is not None and seat1.get("score", 0.5) is not None:
            p0_score = float(seat0.get("score", 0.5))
            p1_score = float(seat1.get("score", 0.5))
            if abs(p0_score - p1_score) > 0.3 or p0_score < 0.2 or p1_score < 0.2:
                classifications.add("seat_or_opening_artifact")

        if dup_traj > games * 0.5:
            classifications.add("seat_or_opening_artifact")

    scores_by_chall_sims: dict[int, list[float]] = {}
    for row in results:
        if "error" in row or "score" not in row:
            continue
        cs = row["challenger_simulations"]
        cur_s = row["current_simulations"]
        score = row["score"]
        if cur_s == 256 and cs != cur_s:
            scores_by_chall_sims.setdefault(cs, []).append(score)

    if scores_by_chall_sims:
        avg_by_cs = {
            cs: statistics.fmean(scores) for cs, scores in scores_by_chall_sims.items()
        }
        sorted_cs = sorted(avg_by_cs.keys())
        if len(sorted_cs) >= 2:
            monotonically_improving = all(
                avg_by_cs[sorted_cs[i]] <= avg_by_cs[sorted_cs[i + 1]]
                for i in range(len(sorted_cs) - 1)
            )
            any_above_55 = any(s > 0.55 for s in avg_by_cs.values())
            if monotonically_improving or any_above_55:
                classifications.add("search_budget_limiting")

    all_scores = [
        row["score"] for row in results if "error" not in row and "score" in row
    ]
    all_50 = all(abs(s - 0.50) < 1e-6 for s in all_scores) if all_scores else False
    if all_50:
        if not classifications:
            classifications.add("model_policy_not_better")

    if not classifications:
        classifications.add("evaluation_noise_or_protocol_issue")

    return ", ".join(sorted(classifications))


def write_report(
    results: list[dict], raw_policy: dict, classification: str, args: argparse.Namespace
) -> str:
    lines: list[str] = []
    lines.append("# AlphaZero-Lite Arena Ceiling Diagnostic Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append(f"**{classification.upper()}**")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Artifact | Path | SHA256 |")
    lines.append("|----------|------|--------|")
    current_path = Path(args.current)
    lines.append(
        f"| current | {args.current} | `{sha256_file(current_path / 'weights.json')}` |"
    )
    seen_candidates: set[str] = set()
    for row in results:
        if "error" in row:
            continue
        if "candidate_sha256" in row and row["candidate"] not in seen_candidates:
            seen_candidates.add(row["candidate"])
            lines.append(
                f"| {row['candidate']} | {row['candidate_path']} | `{row['candidate_sha256']}` |"
            )
    lines.append("")

    lines.append("## Arena Results by Candidate and Budget Pair")
    lines.append("")
    lines.append(
        "| Candidate | Chall Sims | Curr Sims | Games | Wins | Losses | Draws | Score | CI95 Lower | CI95 Upper | P0 Score | P1 Score | Margin Mean | Game Len Mean | Dup Traj | Move Time ms | Move Time p95 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for row in results:
        if "error" in row:
            lines.append(
                f"| {row['candidate']} | {row['challenger_simulations']} | {row['current_simulations']} | — | — | — | — | — | — | — | — | — | — | — | — | — | ERROR: {row['error']} |"
            )
            continue
        ci95 = row.get("ci95", {})
        seat0 = row.get("challenger_starts_0", {})
        seat1 = row.get("challenger_starts_1", {})
        lines.append(
            f"| {row['candidate']} | {row['challenger_simulations']} | {row['current_simulations']} | {row['games']} | {row['wins']} | {row['losses']} | {row['draws']} | {row['score']:.4f} | {ci95.get('lower', 0):.4f} | {ci95.get('upper', 0):.4f} | {seat0.get('score', 0):.4f} | {seat1.get('score', 0):.4f} | {row.get('margin_mean', 0):.1f} | {row.get('game_length_mean', 0):.1f} | {row.get('duplicate_trajectory_count', 0)} | {row.get('move_time_mean_ms', 0):.1f} | {row.get('move_time_p95_ms', 0):.1f} |"
        )

        ext = row.get("extended")
        if ext:
            ext_ci95 = ext.get("ci95", {})
            lines.append(
                f"| {row['candidate']} (ext) | {row['challenger_simulations']} | {row['current_simulations']} | {ext['games']} | {ext['wins']} | {ext['losses']} | {ext['draws']} | {ext['score']:.4f} | {ext_ci95.get('lower', 0):.4f} | {ext_ci95.get('upper', 0):.4f} | — | — | — | — | — | {ext.get('move_time_mean_ms', 0):.1f} | {ext.get('move_time_p95_ms', 0):.1f} |"
            )
    lines.append("")

    # Forced seat-split table
    lines.append("## Forced Seat-Split Results")
    lines.append("")
    lines.append(
        "| Candidate | Chall Sims | Curr Sims | Starts | Games | Wins | Losses | Draws | Score |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in results:
        if "error" in row:
            continue
        for forced_key in ("forced_starts_0", "forced_starts_1"):
            fs = row.get(forced_key)
            if fs is None:
                continue
            starts_label = forced_key.rsplit("_", 1)[-1]
            lines.append(
                f"| {row['candidate']} | {row['challenger_simulations']} | {row['current_simulations']} | {starts_label} | {fs['games']} | {fs['wins']} | {fs['losses']} | {fs['draws']} | {fs['score']:.4f} |"
            )
    lines.append("")

    # First move distributions
    lines.append("## First Move Distributions")
    lines.append("")
    lines.append(
        "| Candidate | Chall Sims | Curr Sims | Challenger First Moves | Current First Moves |"
    )
    lines.append("|---|---|---|---|---|")
    for row in results:
        if "error" in row:
            continue
        fmc = row.get("first_move_challenger_dist", {})
        fmcu = row.get("first_move_current_dist", {})
        lines.append(
            f"| {row['candidate']} | {row['challenger_simulations']} | {row['current_simulations']} | {json.dumps(fmc)} | {json.dumps(fmcu)} |"
        )
    lines.append("")

    # Raw policy diagnostic
    if raw_policy:
        lines.append("## Raw Policy Diagnostic")
        lines.append("")
        lines.append("| Metric | " + " | ".join(raw_policy.keys()) + " |")
        lines.append("|" + "---|" * (len(raw_policy) + 1))
        all_metrics = set()
        for metrics in raw_policy.values():
            all_metrics.update(metrics.keys())
        for metric in sorted(all_metrics):
            vals = []
            for cand_name in raw_policy:
                v = raw_policy[cand_name].get(metric)
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v) if v is not None else "—")
            lines.append(f"| {metric} | " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Classification Rationale")
    lines.append("")
    lines.append(f"Classification: **{classification}**")
    lines.append("")

    # Write per-candidate budget analysis
    lines.append("### Candidate-vs-Current Score by Search Budget (challenger:current)")
    lines.append("")
    lines.append(
        "Candidate | 128:128 | 256:256 | 384:384 | 768:768 | 1200:1200 | 384:256 | 768:256 | 1200:256 | 256:768"
    )
    lines.append("---|---|---|---|---|---|---|---|---|---")
    for cand_name in sorted(
        set(row["candidate"] for row in results if "error" not in row)
    ):
        scores: dict[str, str] = {}
        for row in results:
            if "error" in row or row["candidate"] != cand_name:
                continue
            pair = f"{row['challenger_simulations']}:{row['current_simulations']}"
            scores[pair] = f"{row['score']:.4f}"
        ordered = [
            "128:128",
            "256:256",
            "384:384",
            "768:768",
            "1200:1200",
            "384:256",
            "768:256",
            "1200:256",
            "256:768",
        ]
        values = [scores.get(p, "—") for p in ordered]
        lines.append(f"| {cand_name} | " + " | ".join(values) + " |")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("=== Arena Ceiling Diagnostic ===", flush=True)
    print(f"Current: {args.current}", flush=True)
    print(f"Candidates: {args.candidates}", flush=True)
    print(f"Budget pairs: {args.budget_pairs}", flush=True)
    print(f"Games: {args.games}, Extended: {args.extended_games}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print("", flush=True)

    results = run_all_arenas(args)

    raw_policy = {}
    if not args.skip_raw_policy:
        print("\n=== Raw Policy Diagnostic ===", flush=True)
        raw_policy = run_raw_policy_diagnostic(args)

    classification = classify_results(results, raw_policy)

    report = write_report(results, raw_policy, classification, args)

    out_path = args.out_report or str(workdir / "arena_ceiling_diagnostic_report.md")
    if not args.out_report:
        docs_path = (
            REPO_ROOT / "docs/alphazero-lite-arena-ceiling-diagnostic-results.md"
        )
        Path(docs_path).write_text(report, encoding="utf-8")
        print(f"\nReport written to {docs_path}", flush=True)
    Path(out_path).write_text(report, encoding="utf-8")
    print(f"Classification: {classification}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
