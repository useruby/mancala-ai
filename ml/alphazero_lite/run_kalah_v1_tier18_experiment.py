#!/usr/bin/env python3
"""Generate twice and validate an isolated canonical KVTB1 tier-18 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

from ml.alphazero_lite.exact_kalah_solver import ExactKalahSolver, ExactState
from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import (
    ROOT,
    SAMPLE_SEED,
    cited_gate,
    generate,
    independent_oracle_sample,
    portable_format_gate,
    probe,
    rank,
    state_count,
    unrank,
)

TIER = 18
EXPECTED_STATES = 172_986_450
EXPECTED_PAYLOADS = {
    8: "441341d8825a19ab6126e8194f7045ec36f869173e788bcccce544cfff68ef94",
    12: "d4728322ad902d504a700caef6b74b819453163420e7417f9227e279ad69595a",
}


class KvtbReader:
    """Independent Python KVTB1 decoder used only for artifact validation."""

    def __init__(self, path: Path) -> None:
        self.data = path.read_bytes()
        if self.data[:5] != b"KVTB1" or int.from_bytes(self.data[5:7], "little") != 1:
            raise AssertionError("invalid KVTB1 header")
        if self.data[7:15] != b"kalah_v1" or self.data[15:18] != bytes((1, 1, 128)):
            raise AssertionError("unexpected canonical encoding")
        self.tier = self.data[18]
        self.revision = int.from_bytes(self.data[19:23], "little")
        header_size = 80 + 16 * (self.tier + 1)
        self.states = int.from_bytes(self.data[24:32], "little")
        payload_size = int.from_bytes(self.data[32:40], "little")
        if self.states != payload_size or len(self.data) != header_size + payload_size:
            raise AssertionError("invalid KVTB1 lengths")
        self.offsets = [
            int.from_bytes(self.data[56 + 16 * tier : 64 + 16 * tier], "little")
            for tier in range(self.tier + 1)
        ]
        self.payload = self.data[header_size:]
        if (
            hashlib.sha256(self.payload).digest()
            != self.data[header_size - 32 : header_size]
        ):
            raise AssertionError("payload checksum mismatch")

    def value(self, pits: tuple[int, ...], player: int) -> int:
        tier = sum(pits)
        if tier > self.tier:
            raise AssertionError("lookup above tablebase tier")
        value = self.payload[self.offsets[tier] + 2 * rank(pits) + player]
        if value == 128:
            raise AssertionError("unknown tablebase entry")
        return value - 256 if value > 127 else value


def _complete_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix_gate(reader: KvtbReader) -> dict[str, Any]:
    prefixes = {}
    for tier, expected in EXPECTED_PAYLOADS.items():
        end = sum(state_count(level) for level in range(tier + 1))
        digest = hashlib.sha256(reader.payload[:end]).hexdigest()
        if digest != expected:
            raise AssertionError(f"tier-{tier} payload prefix mismatch")
        prefixes[str(tier)] = digest
    return {"passed": True, "payload_sha256": prefixes}


def _tier8_gate(binary: Path, tablebase: Path) -> dict[str, Any]:
    requests = [
        {"pits": unrank(tier, index), "player": player}
        for tier in range(9)
        for index in range(comb(tier + 11, 11))
        for player in (0, 1)
    ]
    answers = probe(binary, tablebase, requests)
    solver = ExactKalahSolver(tt_size=2_000_000)
    try:
        for request, answer in zip(requests, answers, strict=True):
            state = ExactState(tuple(request["pits"]), (0, 0), request["player"])
            if answer["value"] != solver.solve(state):
                raise AssertionError("tier-8 root mismatch")
            if {
                int(k): v for k, v in answer["actions"].items()
            } != solver.action_margins(state):
                raise AssertionError("tier-8 action mismatch")
    finally:
        solver.close()
    return {"passed": True, "states": len(requests)}


def _sample_9_12_gate(binary: Path, tablebase: Path) -> dict[str, Any]:
    requests = [
        {"pits": pits, "player": player} for pits, player in independent_oracle_sample()
    ]
    answers = probe(binary, tablebase, requests)
    solver = ExactKalahSolver(tt_size=2_000_000)
    try:
        for request, answer in zip(requests, answers, strict=True):
            state = ExactState(tuple(request["pits"]), (0, 0), request["player"])
            if answer["value"] != solver.solve(state):
                raise AssertionError("tier-9--12 sample root mismatch")
    finally:
        solver.close()
    return {"passed": True, "seed": SAMPLE_SEED, "states": len(requests)}


def _stratified_bellman_gate(
    reader: KvtbReader, per_tier: int = 1_000
) -> dict[str, Any]:
    value = 18_271
    events = {
        "roots": 0,
        "extra_turns": 0,
        "captures": 0,
        "terminal_sweeps": 0,
        "one_sided": 0,
    }
    for tier in range(13, 19):
        for index in range(per_tier):
            value = (value * 1103515245 + 12345) & 0x7FFFFFFF
            pits = unrank(tier, value % comb(tier + 11, 11))
            player = (value >> 8) & 1
            state = ExactState(pits, (0, 0), player)
            if rank(unrank(tier, rank(pits))) != rank(pits):
                raise AssertionError("independent rank/unrank mismatch")
            if state.is_terminal():
                expected = sum(pits[:6]) - sum(pits[6:])
                if reader.value(pits, player) != expected:
                    raise AssertionError("terminal base-value mismatch")
                events["one_sided"] += 1
                events["roots"] += 1
                continue
            actions = state.legal_moves()
            child_values = []
            for action in actions:
                child = state.play(action)
                delta = child.stores[0] - child.stores[1]
                if child.is_terminal():
                    candidate = delta
                    events["terminal_sweeps"] += 1
                else:
                    candidate = delta + reader.value(child.pits, child.current_player)
                child_values.append(candidate)
                events["extra_turns"] += int(
                    child.current_player == player and not child.is_terminal()
                )
                events["captures"] += int(abs(delta) > 1)
            expected = max(child_values) if player == 0 else min(child_values)
            if reader.value(pits, player) != expected:
                raise AssertionError("Bellman recurrence mismatch")
            events["one_sided"] += int(not any(pits[:6]) or not any(pits[6:]))
            events["roots"] += 1
    if not all(
        events[name]
        for name in ("extra_turns", "captures", "terminal_sweeps", "one_sided")
    ):
        raise AssertionError(f"stratified sample missed required event: {events}")
    return {"passed": True, "per_tier": per_tier, "events": events}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    binary = Path(
        subprocess.check_output(
            ["bash", "native/kalah_v1_tablebase/build.sh"], cwd=ROOT, text=True
        ).strip()
    )
    result: dict[str, Any] = {
        "schema": "kalah_v1_tier18_experiment_v1",
        "training_eligible": False,
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "compiler": subprocess.check_output(
            ["c++", "--version"], text=True
        ).splitlines()[0],
        "gates": {},
    }
    directories = [
        Path(tempfile.mkdtemp(prefix="kalah_v1_tier18_", dir=args.temporary_root))
        for _ in range(2)
    ]
    try:
        paths = [directory / "kalah_v1_18.kvtb" for directory in directories]
        runs = [
            generate(
                binary,
                TIER,
                path,
                memory_limit_kib=16 * 1024 * 1024,
                timeout_seconds=8 * 60 * 60,
            )
            for path in paths
        ]
        for run, path in zip(runs, paths, strict=True):
            run["complete_file_sha256"] = _complete_sha(path)
            if (
                run["states"] != EXPECTED_STATES
                or run["cycles"]
                or run["output_bytes"] > 4 * 1024**3
            ):
                raise AssertionError("tier-18 generation requirements failed")
        identical = all(
            runs[0][key] == runs[1][key]
            for key in (
                "payload_sha256",
                "complete_file_sha256",
                "output_bytes",
                "states",
                "edges",
            )
        )
        if not identical:
            raise AssertionError("fresh tier-18 generations differ")
        result["generation"] = {"passed": True, "runs": runs, "identical": True}
        reader = KvtbReader(paths[0])
        if reader.tier != TIER or reader.states != EXPECTED_STATES:
            raise AssertionError("tier-18 metadata mismatch")
        result["gates"]["portable_format"] = portable_format_gate(
            binary, paths[0], directories[0]
        )
        result["gates"]["validated_prefixes"] = _prefix_gate(reader)
        result["gates"]["cited_positions"] = cited_gate(binary, paths[0])
        result["gates"]["complete_tier_8_oracle"] = _tier8_gate(binary, paths[0])
        result["gates"]["tier_9_12_oracle_sample"] = _sample_9_12_gate(binary, paths[0])
        result["gates"]["tier_13_18_bellman"] = _stratified_bellman_gate(reader)
        result["artifact_path"] = str(paths[0])
        result["generator_revision"] = reader.revision
        result["classification"] = "canonical_18_tablebase_validated"
    except Exception as error:
        result["classification"] = (
            "canonical_18_tablebase_budget_exceeded"
            if "limit" in str(error)
            else "canonical_18_tablebase_incorrect"
        )
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["finished_unix_seconds"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["classification"] == "canonical_18_tablebase_validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
