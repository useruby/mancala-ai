#!/usr/bin/env python3
"""Differential fuzzing between Python and Ruby Kalah engines."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.kalah_rules import KalahGame


RUBY_TRACE_CODE = """
require 'json'
payload = JSON.parse(File.read(ARGV.fetch(0)))
game = Games::Kalah.from_state(payload.fetch('initial_state'))
steps = []
payload.fetch('moves').each do |relative_move|
  absolute_move = game.pit_index(relative_move)
  ok = game.move(absolute_move)
  steps << {
    'relative_move' => relative_move,
    'absolute_move' => absolute_move,
    'ok' => ok,
    'state' => game.to_state,
    'winner' => game.winner,
    'over' => game.over?,
    'possible_moves' => game.possible_moves
  }
end
puts JSON.generate({ 'steps' => steps })
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--out", default="/tmp/kalah_parity_fuzz_report.json")
    return parser.parse_args()


def generate_python_trace(rng: random.Random, max_moves: int) -> dict:
    game = KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    moves = []
    steps = []

    for _ in range(max_moves):
        if game.over():
            break

        legal_moves = game.possible_moves()
        if not legal_moves:
            break

        relative_move = rng.choice(legal_moves)
        absolute_move = game.pit_index(relative_move)
        ok = game.move(absolute_move)
        moves.append(relative_move)
        steps.append(
            {
                "relative_move": relative_move,
                "absolute_move": absolute_move,
                "ok": ok,
                "state": game.to_state(),
                "winner": game.winner,
                "over": game.over(),
                "possible_moves": game.possible_moves(),
            }
        )

    return {
        "initial_state": {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        "moves": moves,
        "steps": steps,
    }


def ruby_trace_for(initial_state: dict, moves: list[int]) -> list[dict]:
    payload = {"initial_state": initial_state, "moves": moves}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle)
        payload_path = handle.name

    command = ["bin/rails", "runner", RUBY_TRACE_CODE, payload_path]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    Path(payload_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ruby trace failed: {result.stderr.strip()}")

    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)["steps"]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    mismatches = []

    for game_index in range(args.games):
        python_trace = generate_python_trace(rng=rng, max_moves=args.max_moves)
        ruby_steps = ruby_trace_for(
            initial_state=python_trace["initial_state"],
            moves=python_trace["moves"],
        )

        python_steps = python_trace["steps"]
        if len(python_steps) != len(ruby_steps):
            mismatches.append(
                {
                    "game_index": game_index,
                    "reason": "step_count_mismatch",
                    "python_steps": len(python_steps),
                    "ruby_steps": len(ruby_steps),
                }
            )
            continue

        for step_index, (py_step, rb_step) in enumerate(
            zip(python_steps, ruby_steps, strict=True)
        ):
            if py_step != rb_step:
                mismatches.append(
                    {
                        "game_index": game_index,
                        "step_index": step_index,
                        "reason": "step_mismatch",
                        "python_step": py_step,
                        "ruby_step": rb_step,
                    }
                )
                break

    report = {
        "schema": "kalah_parity_fuzz_v1",
        "seed": args.seed,
        "games": args.games,
        "max_moves": args.max_moves,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "parity_passed": len(mismatches) == 0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote parity fuzz report to {out_path}")
    print(
        f"parity_passed={report['parity_passed']} mismatch_count={report['mismatch_count']}"
    )


if __name__ == "__main__":
    main()
