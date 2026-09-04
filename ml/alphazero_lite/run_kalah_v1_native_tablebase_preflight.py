#!/usr/bin/env python3
# ruff: noqa: E402
"""Reproducible diagnostic validation for the isolated native kalah_v1 tablebase."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ml.alphazero_lite.exact_kalah_solver import ExactKalahSolver, ExactState
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state

SAMPLE_SEED = 277
SAMPLE_COUNT = 10_000


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
    peak_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = json.loads(completed.stdout)
    result.update(
        wall_seconds=time.perf_counter() - started,
        peak_rss_kib=peak_rss_kib,
        output_bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    result["bytes_per_indexed_state"] = result["output_bytes"] / result["states"]
    return result


def probe(
    binary: Path, tablebase: Path, requests: list[dict], *, transition: bool = False
) -> list[dict]:
    command = [binary] if transition else [binary, "probe", tablebase]
    process = subprocess.run(
        command,
        input="".join(json.dumps(item) + "\n" for item in requests),
        check=True,
        capture_output=True,
        text=True,
    )
    answers = [json.loads(line) for line in process.stdout.splitlines()]
    if len(answers) != len(requests) or any("error" in answer for answer in answers):
        raise AssertionError("native returned an unknown entry or illegal transition")
    return answers


def rank_gate(limit: int = 10) -> dict:
    states, one_sided = 0, {0: 0, 1: 0}
    for tier in range(limit + 1):
        for index in range(comb(tier + 11, 11)):
            pits = unrank(tier, index)
            assert rank(pits) == index and unrank(tier, rank(pits)) == pits
            for player in (0, 1):
                states += 1
                if sum(pits[player * 6 : player * 6 + 6]) == tier:
                    one_sided[player] += 1
    assert states == cumulative_count(limit)
    return {"states": states, "one_sided": one_sided, "passed": True}


def _game_state(
    pits: tuple[int, ...], player: int, stores: tuple[int, int] = (0, 0)
) -> dict:
    return {
        "player_pits": list(pits[:6]),
        "opponent_pits": list(pits[6:]),
        "player_store": stores[0],
        "opponent_store": stores[1],
        "current_player": player,
    }


def transition_gate(binary: Path, limit: int) -> dict:
    requests = [
        {"pits": pits, "player": player, "move": move}
        for tier in range(limit + 1)
        for pits in compositions(tier)
        for player in (0, 1)
        for move in range(6)
        if pits[player * 6 + move]
    ]
    answers = probe(binary, Path(), requests, transition=True)
    for request, native in zip(requests, answers, strict=True):
        pits, player, move = tuple(request["pits"]), request["player"], request["move"]
        before = _game_state(pits, player)
        raw = KalahGame.from_state(before)
        absolute = player * 6 + move
        seeds = raw.pits[absolute]
        raw.pits[absolute] = 0
        landing, raw_extra = raw._seeding(absolute, seeds, player)
        if not raw_extra:
            raw._capture(landing)
            raw.current_player = 1 - player
        raw_terminal = raw.over()
        sweep = 0
        if raw_terminal:
            opposite = 1 - raw.current_player
            swept = sum(raw.pits[opposite * 6 : opposite * 6 + 6])
            sweep = swept if opposite == 0 else -swept
        game = KalahGame.from_state(before)
        assert game.move(player * 6 + move)
        exact = ExactState(pits, (0, 0), player).play(move)
        consequence = move_consequence_for_state(before, move)
        assert (
            tuple(game.pits) == exact.pits
            and tuple(game.captured_seeds) == exact.stores
            and game.current_player == exact.current_player
        )
        assert (
            native["pits"] == list(game.pits)
            and native["player"] == game.current_player
        )
        assert native["delta"] == game.captured_seeds[0] - game.captured_seeds[1]
        assert native["capture"] == consequence["capture_count"]
        assert native["terminal"] == raw_terminal == game.over()
        assert native["extra"] == (
            consequence["gives_extra_turn"] and not native["terminal"]
        )
        assert native["sweep"] == sweep
        assert not (native["terminal"] and native["extra"])
    return {
        "states": sum(state_count(t) for t in range(limit + 1)),
        "legal_actions": len(requests),
        "passed": True,
    }


def exact_oracle_gate(binary: Path, tablebase: Path, limit: int) -> dict:
    requests = [
        {"pits": pits, "player": player}
        for tier in range(limit + 1)
        for pits in compositions(tier)
        for player in (0, 1)
    ]
    answers = probe(binary, tablebase, requests)
    solver, actions = ExactKalahSolver(tt_size=2_000_000), 0
    for request, answer in zip(requests, answers, strict=True):
        state = ExactState(tuple(request["pits"]), (0, 0), request["player"])
        assert answer["value"] == solver.solve(state)
        expected = solver.action_margins(state)
        assert {
            int(move): value for move, value in answer["actions"].items()
        } == expected
        actions += len(expected)
    return {"states": len(requests), "actions": actions, "passed": True}


def sample_gate(binary: Path, tablebase: Path) -> dict:
    requests: list[dict] = []
    value = SAMPLE_SEED
    while len(requests) < SAMPLE_COUNT:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        tier = 9 + value % 4
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        requests.append(
            {"pits": unrank(tier, value % comb(tier + 11, 11)), "player": value & 1}
        )
    answers = probe(binary, tablebase, requests)
    solver = ExactKalahSolver(tt_size=2_000_000, cache_enabled=False)
    for request, answer in zip(requests, answers, strict=True):
        state = ExactState(tuple(request["pits"]), (0, 0), request["player"])
        assert answer["value"] == solver.solve(state)
        assert {
            int(move): score for move, score in answer["actions"].items()
        } == solver.action_margins(state)
    return {"seed": SAMPLE_SEED, "query_count": len(requests), "passed": True}


def store_offset_gate(binary: Path, tablebase: Path, limit: int) -> dict:
    requests = [
        {"pits": pits, "player": player}
        for tier in range(limit + 1)
        for pits in compositions(tier)
        for player in (0, 1)
    ]
    answers = probe(binary, tablebase, requests)
    for request, answer in zip(requests, answers, strict=True):
        for stores in ((18, 24), (31, 7)):
            margin = stores[0] - stores[1]
            state = ExactState(tuple(request["pits"]), stores, request["player"])
            solver = ExactKalahSolver(tt_size=100_000)
            assert solver.solve(state) == answer["value"] + margin
            expected = solver.action_margins(state)
            adjusted = {
                int(move): value + margin for move, value in answer["actions"].items()
            }
            assert expected == adjusted
            assert sorted(
                expected, key=expected.get, reverse=request["player"] == 0
            ) == sorted(adjusted, key=adjusted.get, reverse=request["player"] == 0)
    return {"states": len(requests), "margins": [[18, 24], [31, 7]], "passed": True}


def cited_gate(binary: Path, tablebase: Path) -> dict:
    requests = [
        {"pits": [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], "player": 0},
        {"pits": [0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 1], "player": 1},
    ]
    first, second = probe(binary, tablebase, requests)
    assert first["actions"] == {"1": -4, "2": 0}
    assert second["actions"] == {"0": -4, "5": -4}
    assert {move: score - 6 for move, score in second["actions"].items()} == {
        "0": -10,
        "5": -10,
    }
    return {"passed": True}


def determinism_gate(binary: Path, tier: int, directory: Path) -> dict:
    first, second = directory / f"{tier}-a.kvtb", directory / f"{tier}-b.kvtb"
    a, b = generate(binary, tier, first), generate(binary, tier, second)
    assert first.read_bytes() == second.read_bytes()
    return {
        "tier": tier,
        "sha256": a["sha256"],
        "bytes": a["output_bytes"],
        "passed": True,
        "runs": [a, b],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="run exhaustive semantic gates and 9-12 sample oracle",
    )
    args = parser.parse_args()
    if not 0 <= args.tier <= 20:
        raise SystemExit("tier must be 0..20")
    binary = build()
    result: dict[str, object] = {
        "classification": "canonical_tablebase_validation_incomplete",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "cpu_count": os.cpu_count(),
        },
        "gates": {"rank_unrank": rank_gate()},
    }
    with tempfile.TemporaryDirectory(
        prefix="kalah_v1_tablebase_", dir="/tmp"
    ) as temporary:
        directory = Path(temporary)
        tablebase = directory / f"kalah_v1_{args.tier}.kvtb"
        result["generation"] = generate(binary, args.tier, tablebase)
        corrupt = directory / "corrupt.kvtb"
        corrupt.write_bytes(tablebase.read_bytes() + b"trailing")
        rejected = (
            subprocess.run(
                [binary, "probe", corrupt], input="", capture_output=True
            ).returncode
            != 0
        )
        if not rejected:
            raise AssertionError("native reader accepted trailing data")
        result["gates"]["portable_format"] = {
            "passed": True,
            "trailing_data_rejected": True,
        }
        result["gates"]["cited_positions"] = cited_gate(binary, tablebase)
        if args.full_validation:
            if args.tier < 12:
                raise SystemExit("full validation requires --tier 12")
            result["gates"]["transition_parity"] = transition_gate(binary, 8)
            result["gates"]["root_action_oracle"] = exact_oracle_gate(
                binary, tablebase, 8
            )
            result["gates"]["store_offset_invariance"] = store_offset_gate(
                binary, tablebase, 8
            )
            result["gates"]["sample_oracle"] = sample_gate(binary, tablebase)
            result["gates"]["determinism_8"] = determinism_gate(binary, 8, directory)
            result["gates"]["determinism_12"] = determinism_gate(binary, 12, directory)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
