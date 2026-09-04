#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnostic orchestration for the isolated native kalah_v1 tablebase."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ml.alphazero_lite.exact_kalah_solver import ExactKalahSolver, ExactState


def state_count(tier: int) -> int:
    return 2 * comb(tier + 11, 11)


def cumulative_count(tier: int) -> int:
    return 2 * comb(tier + 12, 12)


def compositions(total: int, slots: int = 12):
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, slots - 1):
            yield (first,) + tail


def rank(pits: tuple[int, ...]) -> int:
    result, remaining = 0, sum(pits)
    for index, value in enumerate(pits[:-1]):
        for candidate in range(value):
            result += comb(remaining - candidate + 10 - index, 10 - index)
        remaining -= value
    return result


def unrank(total: int, index: int) -> tuple[int, ...]:
    pits, remaining = [], total
    for position in range(11):
        for value in range(remaining + 1):
            block = comb(remaining - value + 10 - position, 10 - position)
            if index < block:
                pits.append(value)
                remaining -= value
                break
            index -= block
    return tuple(pits + [remaining])


def build() -> Path:
    return Path(
        subprocess.check_output(
            ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
        ).strip()
    )


def generate(binary: Path, tier: int, output: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [binary, "generate", str(tier), output],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    result["wall_seconds"] = time.perf_counter() - started
    result["output_bytes"] = output.stat().st_size
    result["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def probe(binary: Path, tablebase: Path, requests: list[dict]) -> list[dict]:
    process = subprocess.run(
        [binary, "probe", tablebase],
        input="".join(json.dumps(item) + "\n" for item in requests),
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines()]


def rank_gate(limit: int) -> dict:
    states = 0
    one_sided = {0: 0, 1: 0}
    for tier in range(limit + 1):
        assert state_count(tier) == 2 * comb(tier + 11, 11)
        for index in range(comb(tier + 11, 11)):
            pits = unrank(tier, index)
            assert rank(pits) == index
            assert unrank(tier, rank(pits)) == pits
            for player in (0, 1):
                states += 1
                if sum(pits[player * 6 : player * 6 + 6]) == tier:
                    one_sided[player] += 1
    assert states == cumulative_count(limit)
    return {"states": states, "one_sided": one_sided, "passed": True}


def exact_oracle_gate(binary: Path, tablebase: Path, limit: int) -> dict:
    """Exhaustively compare every native root and action value with ExactState."""
    requests = [
        {"pits": pits, "player": player}
        for tier in range(limit + 1)
        for pits in compositions(tier)
        for player in (0, 1)
    ]
    answers = probe(binary, tablebase, requests)
    solver = ExactKalahSolver(tt_size=2_000_000)
    actions = 0
    for request, answer in zip(requests, answers, strict=True):
        state = ExactState(tuple(request["pits"]), (0, 0), request["player"])
        assert answer["value"] == solver.solve(state)
        expected_actions = solver.action_margins(state)
        assert {
            int(move): value for move, value in answer["actions"].items()
        } == expected_actions
        actions += len(expected_actions)
    return {"states": len(requests), "actions": actions, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-oracle", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.tier <= 20:
        raise SystemExit("tier must be 0..20")
    binary = build()
    with tempfile.TemporaryDirectory(
        prefix="kalah_v1_tablebase_", dir="/tmp"
    ) as temporary:
        tablebase = Path(temporary) / f"kalah_v1_{args.tier}.kvtb"
        result: dict[str, object] = {
            "rank_unrank": rank_gate(min(args.tier, 10)),
            "generation": generate(binary, args.tier, tablebase),
        }
        if args.validate_oracle:
            result["exact_oracle"] = exact_oracle_gate(binary, tablebase, args.tier)
    result["state_count_table"] = [
        {"tier": tier, "exact": state_count(tier), "cumulative": cumulative_count(tier)}
        for tier in range(args.tier + 1)
    ]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
