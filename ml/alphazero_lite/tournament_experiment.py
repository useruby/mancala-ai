#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.tournament_decision import pick_best_topk, summarize_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-play-data", required=True)
    parser.add_argument("--bootstrap-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--screen-games", type=int, default=10)
    parser.add_argument("--confirm-games", type=int, default=30)
    parser.add_argument("--challenger-sims", type=int, default=640)
    parser.add_argument("--current-sims", type=int, default=384)
    parser.add_argument("--mcts-sims", type=int, default=1200)
    parser.add_argument("--min-mcts-score", type=float, default=0.45)
    parser.add_argument("--min-arena-score", type=float, default=0.55)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def parse_arena_score(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wins = int(payload["wins"])
    draws = int(payload["draws"])
    games_played = int(payload["games_played"])
    return (wins + (0.5 * draws)) / games_played


def parse_mcts_score(stdout: str) -> float:
    payload = json.loads(stdout)
    return float(payload["score"])


def mcts_baseline_command(
    *,
    challenger_path: Path,
    games: int,
    az_base_simulations: int,
    mcts_simulations: int,
    out_path: Path,
) -> list[str]:
    return [
        ".venv/bin/python",
        "ml/alphazero_lite/mcts1200_baseline.py",
        "--challenger-path",
        str(challenger_path),
        "--games",
        str(games),
        "--az-base-simulations",
        str(az_base_simulations),
        "--mcts-simulations",
        str(mcts_simulations),
        "--workers",
        "6",
        "--root-policy-mode",
        "visit_count",
        "--tactical-root-bias",
        "0.0",
        "--out",
        str(out_path),
    ]


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    winners: list[dict] = []

    for seed in seeds:
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "checkpoint.npz"

        run(
            [
                ".venv/bin/python",
                "ml/alphazero_lite/train.py",
                "--data",
                args.self_play_data,
                "--data-files",
                f"{args.self_play_data},{args.bootstrap_data}",
                "--replay-weights",
                "1,4",
                "--out",
                str(checkpoint),
                "--epochs",
                "14",
                "--batch-size",
                "512",
                "--device",
                "auto",
                "--hidden-sizes",
                "192,192,96",
                "--model-type",
                "mlp_deep",
                "--value-loss",
                "huber",
                "--huber-delta",
                "1.0",
                "--value-loss-weight",
                "0.3",
                "--val-split",
                "0.1",
                "--grad-clip",
                "1.0",
                "--save-top-k",
                str(args.top_k),
                "--top-k-dir",
                str(seed_dir),
                "--seed",
                str(seed),
            ]
        )

        candidates: list[dict] = []
        for topk in range(1, args.top_k + 1):
            topk_npz = seed_dir / f"checkpoint.top{topk}.npz"
            artifact_dir = seed_dir / f"top{topk}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(topk_npz),
                    "--out-dir",
                    str(artifact_dir),
                    "--version",
                    f"seed-{seed}-top{topk}",
                ]
            )

            screen_report_path = seed_dir / f"top{topk}-mcts-screen.json"
            run(
                mcts_baseline_command(
                    challenger_path=artifact_dir,
                    games=args.screen_games,
                    az_base_simulations=args.challenger_sims,
                    mcts_simulations=args.mcts_sims,
                    out_path=screen_report_path,
                )
            )
            mcts_stdout = screen_report_path.read_text(encoding="utf-8")
            mcts_screen_score = parse_mcts_score(mcts_stdout)
            candidates.append(
                {
                    "seed": seed,
                    "topk_index": topk,
                    "artifact_dir": str(artifact_dir),
                    "mcts_screen_score": mcts_screen_score,
                }
            )

        winner = pick_best_topk(candidates)
        winner_dir = winner["artifact_dir"]

        arena_path = out_dir / f"seed-{seed}-arena-confirm.json"
        run(
            [
                ".venv/bin/python",
                "ml/alphazero_lite/arena.py",
                "--challenger",
                winner_dir,
                "--current",
                "model-artifact/current",
                "--games",
                str(args.confirm_games),
                "--challenger-simulations",
                str(args.challenger_sims),
                "--current-simulations",
                str(args.current_sims),
                "--workers",
                "6",
                "--out",
                str(arena_path),
                "--min-score",
                str(args.min_arena_score),
            ]
        )
        arena_score = parse_arena_score(arena_path)

        confirm_report_path = out_dir / f"seed-{seed}-mcts-confirm.json"
        run(
            mcts_baseline_command(
                challenger_path=Path(winner_dir),
                games=args.confirm_games,
                az_base_simulations=args.challenger_sims,
                mcts_simulations=args.mcts_sims,
                out_path=confirm_report_path,
            )
        )
        mcts_confirm_stdout = confirm_report_path.read_text(encoding="utf-8")
        mcts_confirm_score = parse_mcts_score(mcts_confirm_stdout)

        winners.append(
            {
                "seed": seed,
                "topk_index": winner["topk_index"],
                "artifact_dir": winner_dir,
                "mcts_screen_score": winner["mcts_screen_score"],
                "mcts_confirm_score": mcts_confirm_score,
                "arena_score": arena_score,
            }
        )

    summary = summarize_tournament(
        winners,
        min_mcts_score=args.min_mcts_score,
        min_arena_score=args.min_arena_score,
    )

    report = {
        "schema": "azlite_tournament_v1",
        "seeds": seeds,
        "top_k": args.top_k,
        "screen_games": args.screen_games,
        "confirm_games": args.confirm_games,
        "challenger_simulations": args.challenger_sims,
        "current_simulations": args.current_sims,
        "mcts_simulations": args.mcts_sims,
        "seed_winners": winners,
        "summary": summary,
    }

    report_path = out_dir / "tournament_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote tournament report to {report_path}")


if __name__ == "__main__":
    main()
