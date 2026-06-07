#!/usr/bin/env python3
"""Diagnose the 0.50 arena ceiling with search-budget and seat-split analysis.

Does NOT train, promote, or overwrite any model artifacts.
Reads existing artifacts and replay data; runs arena subprocesses only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
_ORIGINAL_WD = os.getcwd()

try:
    import numpy as np
except ModuleNotFoundError:
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(artifact_dir: Path) -> str:
    return sha256_file(artifact_dir / "weights.json")


def wilson_interval_95(score: float, sample_size: int) -> dict:
    from math import sqrt

    if sample_size <= 0:
        return {"lower": 0.0, "upper": 1.0, "margin": 0.0}
    z = 1.96
    n = float(sample_size)
    p = score
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return {
        "lower": round(max(0.0, center - margin), 4),
        "upper": round(min(1.0, center + margin), 4),
        "margin": round(margin, 4),
    }


def run_arena_subprocess(
    *,
    challenger: Path,
    current: Path,
    games: int,
    challenger_sims: int,
    current_sims: int,
    out: Path,
    seed: int,
    workers: int = 8,
    challenger_starts: int | None = None,
    game_jsonl: Path | None = None,
) -> dict:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "ml" / "alphazero_lite" / "arena.py"),
        "--challenger",
        str(challenger),
        "--current",
        str(current),
        "--games",
        str(games),
        "--challenger-simulations",
        str(challenger_sims),
        "--current-simulations",
        str(current_sims),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--min-score",
        "0.0",
        "--out",
        str(out),
    ]
    if challenger_starts is not None:
        cmd.extend(["--challenger-starts", str(challenger_starts)])
    if game_jsonl is not None:
        cmd.extend(["--game-jsonl", str(game_jsonl)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            print(f"  arena subprocess failed: {stderr}", file=sys.stderr)
        return {}
    if not out.exists():
        print(f"  arena report not written: {out}", file=sys.stderr)
        return {}
    return json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# budget matrix
# ---------------------------------------------------------------------------

BUDGET_PAIRS = [
    (128, 128),
    (256, 256),
    (384, 384),
    (768, 768),
    (1200, 1200),
    (384, 256),
    (768, 256),
    (1200, 256),
    (256, 768),
]


def budget_label(cs: int, cu: int) -> str:
    return f"{cs}v{cu}"


# ---------------------------------------------------------------------------
# arena evaluation for a single candidate x budget-pair
# ---------------------------------------------------------------------------


def evaluate_candidate_budget(
    *,
    candidate_label: str,
    challenger_path: Path,
    current_path: Path,
    challenger_sims: int,
    current_sims: int,
    games: int,
    seed: int,
    workers: int,
    workdir: Path,
    challenger_starts: int | None = None,
) -> dict:
    label = budget_label(challenger_sims, current_sims)
    seat_suffix = f"_seat{challenger_starts}" if challenger_starts is not None else ""
    stem = f"{candidate_label}_{label}{seat_suffix}"
    out_path = workdir / f"{stem}_arena.json"
    game_jsonl = workdir / f"{stem}_games.jsonl" if challenger_starts is None else None

    report = run_arena_subprocess(
        challenger=challenger_path,
        current=current_path,
        games=games,
        challenger_sims=challenger_sims,
        current_sims=current_sims,
        out=out_path,
        seed=seed,
        workers=workers,
        challenger_starts=challenger_starts,
        game_jsonl=game_jsonl,
    )
    if not report:
        return {"error": "arena subprocess failed", "budget_pair": label}

    result = {
        "candidate_label": candidate_label,
        "challenger_artifact_sha256": artifact_sha256(challenger_path),
        "current_artifact_sha256": artifact_sha256(current_path),
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "games": int(report.get("games_played", 0)),
        "wins": int(report.get("wins", 0)),
        "losses": int(report.get("losses", 0)),
        "draws": int(report.get("draws", 0)),
        "score": round(float(report.get("score", 0.0)), 4),
        "confidence_interval_95": report.get("confidence_interval_95", {}),
        "move_time_mean_ms": float(
            report.get("notes", {}).get("move_time_mean_ms", 0.0)
        ),
        "move_time_p95_ms": float(report.get("notes", {}).get("move_time_p95_ms", 0.0)),
    }

    if challenger_starts is not None:
        result["challenger_starts"] = challenger_starts

    if game_jsonl and game_jsonl.exists():
        game_entries = [
            json.loads(line)
            for line in game_jsonl.read_text(encoding="utf-8").strip().splitlines()
            if line.strip()
        ]
        result.update(_game_level_analysis(game_entries))
        game_jsonl.unlink()

    return result


def _game_level_analysis(entries: list[dict]) -> dict:
    if not entries:
        return {}

    margins = [e.get("margin", 0) for e in entries]
    game_lengths = [e.get("game_length", 0) for e in entries]
    trajectories = [e.get("trajectory", "") for e in entries]

    def _safe_median(vals):
        return round(float(np.median(vals)), 2) if vals else 0.0

    def _safe_mean(vals):
        return round(statistics.fmean(vals), 2) if vals else 0.0

    traj_counts = Counter(trajectories)
    unique_trajectories = len(traj_counts)
    duplicate_trajectory_count = len(trajectories) - unique_trajectories

    first_moves_by_side: dict[str, Counter] = {
        "challenger": Counter(),
        "current": Counter(),
    }
    for e in entries:
        fmc = e.get("first_move_challenger")
        fmo = e.get("first_move_current")
        if fmc is not None:
            first_moves_by_side["challenger"][fmc] += 1
        if fmo is not None:
            first_moves_by_side["current"][fmo] += 1

    score_by_seat: dict[int, dict] = {}
    for seat in (0, 1):
        seat_games = [e for e in entries if e.get("challenger_player") == seat]
        if not seat_games:
            continue
        w = sum(1 for e in seat_games if e.get("winner") == "challenger")
        losses_seat = sum(1 for e in seat_games if e.get("winner") == "current")
        draws_seat = sum(1 for e in seat_games if e.get("winner") == "draw")
        score_by_seat[seat] = {
            "games": len(seat_games),
            "wins": w,
            "losses": losses_seat,
            "draws": draws_seat,
            "score": round((w + 0.5 * draws_seat) / len(seat_games), 4)
            if seat_games
            else None,
        }

    return {
        "score_when_challenger_player_0": score_by_seat.get(0),
        "score_when_challenger_player_1": score_by_seat.get(1),
        "first_move_distribution_challenger": dict(
            first_moves_by_side["challenger"].most_common()
        ),
        "first_move_distribution_current": dict(
            first_moves_by_side["current"].most_common()
        ),
        "final_margin_mean": _safe_mean(margins),
        "final_margin_median": _safe_median(margins),
        "final_margin_distribution": dict(Counter(margins).most_common(10)),
        "game_length_mean": _safe_mean(game_lengths),
        "game_length_median": _safe_median(game_lengths),
        "game_length_distribution": dict(Counter(game_lengths).most_common(10)),
        "unique_trajectories": unique_trajectories,
        "duplicate_trajectory_count": duplicate_trajectory_count,
        "total_trajectories": len(trajectories),
    }


# ---------------------------------------------------------------------------
# raw-policy move-choice diagnostic
# ---------------------------------------------------------------------------


def load_artifact_evaluator(artifact_dir: Path):
    from ml.alphazero_lite.arena import ArtifactEvaluator

    return ArtifactEvaluator(artifact_dir)


def load_replay_state_rows(filepath: Path, max_rows: int = 500) -> list[dict]:
    rows = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return rows


def run_raw_policy_diagnostic(
    *,
    candidate_path: Path,
    current_path: Path,
    replay_files: list[Path],
    sample_size: int = 500,
) -> dict:
    from ml.alphazero_lite.kalah_rules import KalahGame

    candidate_eval = load_artifact_evaluator(candidate_path)
    current_eval = load_artifact_evaluator(current_path)

    all_rows: list[dict] = []
    for rp in replay_files:
        if rp.exists():
            all_rows.extend(load_replay_state_rows(rp, max_rows=sample_size))

    if len(all_rows) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_rows), size=sample_size, replace=False)
        all_rows = [all_rows[i] for i in indices]

    candidate_entropies: list[float] = []
    current_entropies: list[float] = []
    candidate_values: list[float] = []
    current_values: list[float] = []

    top_move_agreement_c_vs_curr = 0
    top_move_agreement_c_vs_teacher = 0
    top_move_agreement_curr_vs_teacher = 0
    valid_rows = 0

    for row in all_rows:
        state = row.get("state")
        if not isinstance(state, list) or len(state) < 15:
            continue

        player_pits = [max(0, int(round(float(state[i]) * 48.0))) for i in range(6)]
        opponent_pits = [
            max(0, int(round(float(state[i]) * 48.0))) for i in range(6, 12)
        ]
        player_store = max(0, int(round(float(state[12]) * 48.0)))
        opponent_store = max(0, int(round(float(state[13]) * 48.0)))
        current_player = 0 if float(state[14]) < 0.5 else 1

        game_state = {
            "player_pits": player_pits,
            "opponent_pits": opponent_pits,
            "player_store": player_store,
            "opponent_store": opponent_store,
            "current_player": current_player,
        }

        try:
            game = KalahGame.from_state(game_state)
        except Exception:
            continue

        try:
            c_policy, c_value = candidate_eval.evaluate(game.clone())
            cu_policy, cu_value = current_eval.evaluate(game.clone())
        except Exception:
            continue

        legal = game.possible_moves()
        if not legal:
            continue

        valid_rows += 1
        candidate_values.append(float(c_value))
        current_values.append(float(cu_value))

        c_top = int(np.argmax(c_policy[legal]))
        cu_top = int(np.argmax(cu_policy[legal]))

        if c_top == cu_top:
            top_move_agreement_c_vs_curr += 1

        teacher_policy = row.get("policy")
        teacher_top = None
        if isinstance(teacher_policy, list) and len(teacher_policy) > 0:
            try:
                teacher_top = int(
                    max(
                        (i for i in legal if i < len(teacher_policy)),
                        key=lambda i: teacher_policy[i],
                    )
                )
            except (ValueError, TypeError):
                pass
            if teacher_top is not None:
                if c_top == teacher_top:
                    top_move_agreement_c_vs_teacher += 1
                if cu_top == teacher_top:
                    top_move_agreement_curr_vs_teacher += 1

        def _entropy(policy, leg):
            eps = 1e-10
            probs = policy[leg] / (policy[leg].sum() + eps)
            return -float(np.sum(probs * np.log(probs + eps)))

        candidate_entropies.append(_entropy(c_policy, legal))
        current_entropies.append(_entropy(cu_policy, legal))

    if valid_rows == 0:
        return {"error": "no valid states", "sampled_state_count": 0}

    return {
        "sampled_state_count": valid_rows,
        "candidate_current_top_move_agreement": round(
            top_move_agreement_c_vs_curr / valid_rows, 4
        ),
        "candidate_teacher_top_move_agreement": round(
            top_move_agreement_c_vs_teacher / valid_rows, 4
        ),
        "current_teacher_top_move_agreement": round(
            top_move_agreement_curr_vs_teacher / valid_rows, 4
        ),
        "candidate_policy_entropy_mean": round(
            statistics.fmean(candidate_entropies), 4
        ),
        "current_policy_entropy_mean": round(statistics.fmean(current_entropies), 4),
        "candidate_value_mean": round(statistics.fmean(candidate_values), 4),
        "candidate_value_std": round(statistics.pstdev(candidate_values), 4),
        "current_value_mean": round(statistics.fmean(current_values), 4),
        "current_value_std": round(statistics.pstdev(current_values), 4),
    }


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def classify_results(all_results: list[dict]) -> dict:
    classification: dict[str, list[str]] = {
        "seat_or_opening_artifact": [],
        "search_budget_limiting": [],
        "model_policy_not_better": [],
        "evaluation_noise_or_protocol_issue": [],
    }

    for result in all_results:
        if result.get("error"):
            continue
        label = result.get("candidate_label", "?")
        cs = result.get("challenger_simulations", 0)
        cu = result.get("current_simulations", 0)
        score = result.get("score", 0.0)

        # Seat / opening check
        p0 = result.get("score_when_challenger_player_0")
        p1 = result.get("score_when_challenger_player_1")
        if p0 and p1 and isinstance(p0, dict) and isinstance(p1, dict):
            s0 = p0.get("score", 0.5)
            s1 = p1.get("score", 0.5)
            if abs(s0 - s1) > 0.4 and s0 is not None and s1 is not None:
                classification["seat_or_opening_artifact"].append(
                    f"{label} {budget_label(cs, cu)}: seat asymmetry "
                    f"p0={s0:.2f} vs p1={s1:.2f}"
                )
        dup = result.get("duplicate_trajectory_count", 0)
        total = result.get("total_trajectories", 1)
        if total > 0 and dup / total > 0.5:
            classification["seat_or_opening_artifact"].append(
                f"{label} {budget_label(cs, cu)}: {dup}/{total} duplicate trajectories"
            )

        # Search budget check
        if cs > cu and score > 0.55:
            classification["search_budget_limiting"].append(
                f"{label} {budget_label(cs, cu)}: score={score:.3f} > 0.55 with "
                f"challenger_sims={cs} > current_sims={cu}"
            )

        # Model policy not better: all at/below 0.50 across all budget pairs
        # (tracked across the full set below)

    # Equal-budget check for search-budget limiting
    eq_results = [
        r
        for r in all_results
        if not r.get("error")
        and r.get("challenger_simulations") == r.get("current_simulations")
    ]
    std_results = [
        r
        for r in all_results
        if not r.get("error")
        and r.get("challenger_simulations") == 384
        and r.get("current_simulations") == 256
    ]

    for eq in eq_results:
        label = eq.get("candidate_label", "?")
        cs = eq.get("challenger_simulations", 0)
        score = eq.get("score", 0.0)
        for std in std_results:
            if std.get("candidate_label") == label and std.get("score", 0.0) != score:
                classification["search_budget_limiting"].append(
                    f"{label} equal={cs}v{cs}:{score:.3f} != std_384v256:{std['score']:.3f}"
                )

    # Overall classification: model_policy_not_better if all scores <= 0.50
    all_scores = [
        r["score"]
        for r in all_results
        if not r.get("error") and isinstance(r.get("score"), (int, float))
    ]
    if all_scores and all(s <= 0.50 + 1e-6 for s in all_scores):
        classification["model_policy_not_better"].append(
            f"all {len(all_scores)} scores <= 0.50"
        )

    # evaluation_noise check: seed sensitivity or extended disagreement
    # (will be populated after comparing 120 vs 240 game runs)

    return {k: v for k, v in classification.items() if v}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the 0.50 arena ceiling with search-budget and seat-split analysis."
    )
    parser.add_argument(
        "--workdir", required=True, help="Working directory for diagnostic outputs."
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current production artifact directory.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated list of candidate artifact directories.",
    )
    parser.add_argument(
        "--budget-pairs",
        default="128:128,256:256,384:384,768:768,1200:1200,384:256,768:256,1200:256,256:768",
        help="Comma-separated list of challenger:current sim pairs.",
    )
    parser.add_argument("--games", type=int, default=120, help="Games per budget pair.")
    parser.add_argument(
        "--extended-games",
        type=int,
        default=240,
        help="Extended games for near-threshold results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-seat-split",
        action="store_true",
        help="Skip forced seat-split evaluations.",
    )
    parser.add_argument(
        "--skip-raw-policy",
        action="store_true",
        help="Skip raw-policy move-choice diagnostic.",
    )
    parser.add_argument(
        "--replay-files",
        default=None,
        help="Comma-separated replay files for raw-policy diagnostic.",
    )
    parser.add_argument(
        "--policy-sample-size",
        type=int,
        default=500,
        help="States to sample for raw-policy diagnostic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = Path(args.current).resolve()
    candidate_paths = [
        Path(p.strip()).resolve() for p in args.candidates.split(",") if p.strip()
    ]
    budget_pairs = []
    for pair_str in args.budget_pairs.split(","):
        parts = pair_str.strip().split(":")
        if len(parts) == 2:
            budget_pairs.append((int(parts[0]), int(parts[1])))

    # Validate artifacts
    for p in [current_path] + candidate_paths:
        if not (p / "weights.json").exists():
            print(f"ERROR: missing weights.json in {p}", file=sys.stderr)
            return 1

    current_sha = artifact_sha256(current_path)
    print(f"Current artifact: {current_path}  SHA256={current_sha}")

    candidates_info = []
    for cp in candidate_paths:
        sha = artifact_sha256(cp)
        label = cp.parent.name if cp.name == "artifact" else cp.name
        candidates_info.append({"path": cp, "label": label, "sha256": sha})
        print(f"Candidate: {label}  SHA256={sha}  path={cp}")

    # -----------------------------------------------------------------------
    # Phase 1: Arena matrix
    # -----------------------------------------------------------------------
    print("\n=== Phase 1: Arena Budget Matrix ===")
    all_results: list[dict] = []

    for cand in candidates_info:
        for cs, cu in budget_pairs:
            label = budget_label(cs, cu)
            print(f"\n  {cand['label']}  {label}  ({args.games} games)...")
            result = evaluate_candidate_budget(
                candidate_label=cand["label"],
                challenger_path=cand["path"],
                current_path=current_path,
                challenger_sims=cs,
                current_sims=cu,
                games=args.games,
                seed=args.seed,
                workers=args.workers,
                workdir=workdir,
            )
            if result.get("error"):
                print(f"    ERROR: {result['error']}")
                all_results.append(result)
                continue

            score = result["score"]
            print(
                f"    score={score:.4f}  W={result['wins']} L={result['losses']} D={result['draws']}  "
                f"CI95=[{result['confidence_interval_95'].get('lower', '?')}, "
                f"{result['confidence_interval_95'].get('upper', '?')}]"
            )
            all_results.append(result)

            # Extended check if near 0.45-0.55
            if 0.45 <= score <= 0.55 and args.games < args.extended_games:
                print(f"    Near threshold -> extended {args.extended_games} games...")
                extended_seed = args.seed + 1000
                ext_result = evaluate_candidate_budget(
                    candidate_label=cand["label"],
                    challenger_path=cand["path"],
                    current_path=current_path,
                    challenger_sims=cs,
                    current_sims=cu,
                    games=args.extended_games,
                    seed=extended_seed,
                    workers=args.workers,
                    workdir=workdir,
                )
                if not ext_result.get("error"):
                    ext_score = ext_result["score"]
                    print(
                        f"      extended score={ext_score:.4f}  "
                        f"W={ext_result['wins']} L={ext_result['losses']} D={ext_result['draws']}"
                    )
                    ext_result["extended"] = True
                    ext_result["extended_seed"] = extended_seed
                    all_results.append(ext_result)

    # -----------------------------------------------------------------------
    # Phase 2: Forced seat-split
    # -----------------------------------------------------------------------
    if not args.skip_seat_split:
        print("\n=== Phase 2: Forced Seat-Split ===")
        for cand in candidates_info:
            for seat in (0, 1):
                print(
                    f"\n  {cand['label']}  seat={seat} (challenger always player {seat})  "
                    f"384v256  ({args.games} games)..."
                )
                result = evaluate_candidate_budget(
                    candidate_label=cand["label"],
                    challenger_path=cand["path"],
                    current_path=current_path,
                    challenger_sims=384,
                    current_sims=256,
                    games=args.games,
                    seed=args.seed,
                    workers=args.workers,
                    workdir=workdir,
                    challenger_starts=seat,
                )
                if result.get("error"):
                    print(f"    ERROR: {result['error']}")
                    continue
                score = result["score"]
                print(
                    f"    score={score:.4f}  W={result['wins']} L={result['losses']} D={result['draws']}"
                )
                result["forced_seat_split"] = True
                all_results.append(result)

    # -----------------------------------------------------------------------
    # Phase 3: Raw-policy diagnostic
    # -----------------------------------------------------------------------
    policy_result = {}
    if not args.skip_raw_policy:
        print("\n=== Phase 3: Raw-Policy Diagnostic ===")
        replay_files = []
        if args.replay_files:
            replay_files = [
                Path(p.strip()) for p in args.replay_files.split(",") if p.strip()
            ]
        else:
            defaults = [
                Path("/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl"),
                Path(
                    "/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl"
                ),
                Path(
                    "/tmp/azlite_iterative_random_replay/iter1_candidate_random_train.jsonl"
                ),
            ]
            replay_files = [p for p in defaults if p.exists()]

        if replay_files:
            print(f"  Replay files: {[str(r) for r in replay_files]}")
            for cand in candidates_info:
                print(f"\n  {cand['label']} raw-policy vs current...")
                policy_result[cand["label"]] = run_raw_policy_diagnostic(
                    candidate_path=cand["path"],
                    current_path=current_path,
                    replay_files=replay_files,
                    sample_size=args.policy_sample_size,
                )
                pp = policy_result[cand["label"]]
                if pp.get("error"):
                    print(f"    ERROR: {pp['error']}")
                else:
                    print(
                        f"    states={pp['sampled_state_count']}  "
                        f"cand/curr_agree={pp['candidate_current_top_move_agreement']:.3f}  "
                        f"cand/teacher_agree={pp.get('candidate_teacher_top_move_agreement', '?'):.3f}  "
                        f"curr/teacher_agree={pp.get('current_teacher_top_move_agreement', '?'):.3f}  "
                        f"cand_entropy={pp['candidate_policy_entropy_mean']:.3f}  "
                        f"curr_entropy={pp['current_policy_entropy_mean']:.3f}"
                    )
        else:
            print("  No replay files found. Skipping.")

    # -----------------------------------------------------------------------
    # Phase 4: Classification
    # -----------------------------------------------------------------------
    print("\n=== Phase 4: Classification ===")
    classification = classify_results(all_results)

    # Write full report
    report = {
        "schema": "azlite_arena_ceiling_diagnostic_v1",
        "current_path": str(current_path),
        "current_sha256": current_sha,
        "candidates": [
            {"label": c["label"], "path": str(c["path"]), "sha256": c["sha256"]}
            for c in candidates_info
        ],
        "budget_pairs": [budget_label(cs, cu) for cs, cu in budget_pairs],
        "games_per_budget": args.games,
        "extended_games": args.extended_games,
        "seed": args.seed,
        "arena_results": all_results,
        "raw_policy_diagnostic": policy_result,
        "classification": classification,
    }

    out_path = workdir / "arena_ceiling_diagnostic.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report: {out_path}")

    # Summary for docs
    print("\n=== Summary ===")
    print(f"Total arena results: {len([r for r in all_results if not r.get('error')])}")
    print(f"Errors: {len([r for r in all_results if r.get('error')])}")
    print("\nClassification:")
    for category, evidence in classification.items():
        if evidence:
            print(f"  {category}:")
            for item in evidence:
                print(f"    - {item}")
        else:
            print(f"  {category}: (no evidence)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
