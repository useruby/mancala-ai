#!/usr/bin/env python3
# ruff: noqa: E402
"""Reproducible diagnostic validation for the isolated native kalah_v1 tablebase."""

from __future__ import annotations

import argparse
from functools import cache
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
SCALABILITY_TIERS = (8, 10, 12, 14)
GENERATOR_MEMORY_LIMIT_KIB = 8 * 1024 * 1024
GENERATOR_TIMEOUT_SECONDS = 30 * 60
GENERATOR_DISK_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
MAX_BYTES_PER_STATE = 2
PROJECTION_SAFETY_FACTOR = 2


def _limit_generator_resources(memory_limit_kib: int, cpu_limit_seconds: int) -> None:
    memory_limit_bytes = memory_limit_kib * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))


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


def generate(
    binary: Path,
    tier: int,
    output: Path,
    *,
    memory_limit_kib: int = GENERATOR_MEMORY_LIMIT_KIB,
    timeout_seconds: int = GENERATOR_TIMEOUT_SECONDS,
) -> dict:
    started = time.perf_counter()
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(
        [binary, "generate", str(tier), output],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: _limit_generator_resources(
            memory_limit_kib, timeout_seconds
        ),
    )
    deadline = started + timeout_seconds
    peak_rss_kib = 0
    while True:
        try:
            status = Path(f"/proc/{process.pid}/status").read_text()
            high_watermark = next(
                (line for line in status.splitlines() if line.startswith("VmHWM:")),
                None,
            )
            if high_watermark is not None:
                peak_rss_kib = max(peak_rss_kib, int(high_watermark.split()[1]))
        except FileNotFoundError:
            pass
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            process.kill()
            process.communicate()
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        try:
            stdout, stderr = process.communicate(timeout=min(0.05, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, stdout, stderr
        )
    result = json.loads(stdout)
    result.update(
        wall_seconds=time.perf_counter() - started,
        cpu_seconds=(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
            - child_usage_before.ru_utime
            - child_usage_before.ru_stime
        ),
        peak_rss_kib=peak_rss_kib,
        output_bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        resource_limits={
            "address_space_kib": memory_limit_kib,
            "cpu_seconds": timeout_seconds,
            "wall_seconds": timeout_seconds,
        },
    )
    result["bytes_per_indexed_state"] = result["output_bytes"] / result["states"]
    result["temporary_disk_peak_bytes"] = output.stat().st_size
    result["thresholds"] = {
        "wall_seconds": timeout_seconds,
        "peak_rss_kib": memory_limit_kib,
        "temporary_disk_bytes": GENERATOR_DISK_LIMIT_BYTES,
        "bytes_per_indexed_state": MAX_BYTES_PER_STATE,
    }
    if result["temporary_disk_peak_bytes"] > GENERATOR_DISK_LIMIT_BYTES:
        raise RuntimeError("temporary disk limit exceeded")
    if result["bytes_per_indexed_state"] > MAX_BYTES_PER_STATE:
        raise RuntimeError("output bytes per indexed state limit exceeded")
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


@cache
def independent_oracle_sample() -> tuple[tuple[tuple[int, ...], int], ...]:
    requests: list[tuple[tuple[int, ...], int]] = []
    value = SAMPLE_SEED
    while len(requests) < SAMPLE_COUNT:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        tier = 9 + value % 4
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        requests.append((unrank(tier, value % comb(tier + 11, 11)), value & 1))
    return tuple(requests)


def sample_gate(binary: Path, tablebase: Path) -> dict:
    requests = [
        {"pits": pits, "player": player} for pits, player in independent_oracle_sample()
    ]
    answers = probe(binary, tablebase, requests)
    solver = ExactKalahSolver(tt_size=2_000_000)
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
    solvers = {
        stores: ExactKalahSolver(tt_size=2_000_000) for stores in ((18, 24), (31, 7))
    }
    for request, answer in zip(requests, answers, strict=True):
        for stores, solver in solvers.items():
            margin = stores[0] - stores[1]
            state = ExactState(tuple(request["pits"]), stores, request["player"])
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


def determinism_gate(
    binary: Path, tier: int, directory: Path, **generator_limits: int
) -> dict:
    first, second = directory / f"{tier}-a.kvtb", directory / f"{tier}-b.kvtb"
    a, b = (
        generate(binary, tier, first, **generator_limits),
        generate(binary, tier, second, **generator_limits),
    )
    assert first.read_bytes() == second.read_bytes()
    return {
        "tier": tier,
        "sha256": a["sha256"],
        "bytes": a["output_bytes"],
        "passed": True,
        "runs": [a, b],
    }


def portable_format_gate(binary: Path, tablebase: Path, directory: Path) -> dict:
    data = tablebase.read_bytes()
    tier = data[18]
    header_size = 80 + 16 * (tier + 1)

    def altered(offset: int) -> bytes:
        fixture = bytearray(data)
        fixture[offset] ^= 1
        return bytes(fixture)

    fixtures = {
        "empty": b"",
        "truncated": data[:-1],
        "magic": altered(0),
        "schema": altered(5),
        "game_id": altered(7),
        "byte_order": altered(15),
        "player_encoding": altered(16),
        "unknown_value": altered(17),
        "tier": altered(18),
        "generator_revision": altered(19),
        "max_tier": altered(23),
        "state_count": altered(24),
        "payload_length": altered(32),
        "file_length": altered(40),
        "tier_state_count": altered(48),
        "tier_offset": altered(56),
        "checksum": altered(header_size - 1),
        "payload": altered(header_size),
        "trailing_data": data + b"trailing",
    }
    rejected: dict[str, bool] = {}
    for name, fixture in fixtures.items():
        path = directory / f"malformed-{name}.kvtb"
        path.write_bytes(fixture)
        rejected[name] = (
            subprocess.run(
                [binary, "probe", path],
                input="",
                capture_output=True,
                text=True,
                timeout=30,
            ).returncode
            != 0
        )
        path.unlink()
    assert all(rejected.values()), "native reader accepted malformed portable fixture"
    return {"fixtures": rejected, "fixture_count": len(fixtures), "passed": True}


def scalability_gate(binary: Path, directory: Path, **generator_limits: int) -> dict:
    runs = {
        str(tier): generate(
            binary, tier, directory / f"scalability-{tier}.kvtb", **generator_limits
        )
        for tier in SCALABILITY_TIERS
    }
    return {"tiers": list(SCALABILITY_TIERS), "runs": runs, "passed": True}


def lookup_benchmark_gate(binary: Path, tablebase: Path) -> dict:
    """Measure lookup without Python startup or JSON-line serialization."""
    completed = subprocess.run(
        [binary, "lookup-benchmark", tablebase, "100000", str(SAMPLE_SEED)],
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark = json.loads(completed.stdout)
    if "corpus_sha256" not in benchmark:
        raise AssertionError(
            "native benchmark did not hash its serialized query corpus"
        )
    benchmark["thresholds"] = {
        "warm_lookup_count": 100000,
        "warm_median_lookup_ns": 1_000_000,
        "warm_p95_lookup_ns": 1_000_000,
        "warm_throughput_per_second": 100000,
    }
    benchmark["warm_throughput_per_second"] = (
        benchmark["warm_lookup_count"] * 1_000_000_000 / benchmark["warm_lookup_ns"]
    )
    benchmark["passed"] = all(
        benchmark[name] >= threshold
        if name == "warm_throughput_per_second"
        else benchmark[name] <= threshold
        for name, threshold in benchmark["thresholds"].items()
    )
    if not benchmark["passed"]:
        raise RuntimeError("native lookup benchmark threshold failed")
    return benchmark


def projection_gate(scalability: dict) -> dict:
    tier_14 = scalability["runs"]["14"]
    ratio = cumulative_count(18) / cumulative_count(14)
    projection = {
        "safety_factor": PROJECTION_SAFETY_FACTOR,
        "state_ratio": ratio,
        "generation_seconds": tier_14["wall_seconds"]
        * ratio
        * PROJECTION_SAFETY_FACTOR,
        "peak_rss_kib": tier_14["peak_rss_kib"] * ratio * PROJECTION_SAFETY_FACTOR,
        "output_bytes": tier_14["output_bytes"] * ratio * PROJECTION_SAFETY_FACTOR,
    }
    projection["thresholds"] = {
        "generation_seconds": 8 * 60 * 60,
        "peak_rss_kib": 16 * 1024 * 1024,
        "output_bytes": 4 * 1024 * 1024 * 1024,
    }
    projection["passed"] = all(
        projection[name] <= projection["thresholds"][name]
        for name in projection["thresholds"]
    )
    return projection


def _error_details(error: Exception) -> dict[str, str]:
    message = str(error).strip() or error.__class__.__name__
    details = {"type": error.__class__.__name__, "message": message}
    if isinstance(error, subprocess.CalledProcessError):
        details["stdout"] = str(error.stdout or "").strip()
        details["stderr"] = str(error.stderr or "").strip()
    return details


def run_gate(
    gates: dict[str, dict], name: str, function, *args, **kwargs
) -> dict | None:
    try:
        data = function(*args, **kwargs)
    except Exception as error:
        gates[name] = {"status": "failed", "error": _error_details(error)}
        return None
    if isinstance(data, dict) and data.get("passed") is False:
        gates[name] = {
            "status": "failed",
            "data": data,
            "error": {"type": "GateFailed", "message": "gate returned passed=false"},
        }
        return None
    gates[name] = {
        "status": "passed",
        "data": str(data) if isinstance(data, Path) else data,
    }
    return data


def skip_gate(gates: dict[str, dict], name: str, reason: str) -> None:
    gates[name] = {"status": "skipped", "reason": reason}


def classify(gates: dict[str, dict], full_validation: bool) -> tuple[str, bool, str]:
    statuses = {name: gate["status"] for name, gate in gates.items()}
    failed = {name for name, status in statuses.items() if status == "failed"}
    correctness = {
        "rank_unrank",
        "portable_format",
        "cited_positions",
        "transition_parity",
        "root_action_oracle",
        "store_offset_invariance",
        "sample_oracle",
        "determinism_8",
        "determinism_12",
    }
    if failed & correctness:
        return (
            "canonical_tablebase_incorrect",
            False,
            f"correctness gate failed: {sorted(failed & correctness)[0]}",
        )
    generation_error = gates.get("generation", {}).get("error", {})
    generation_text = " ".join(
        str(generation_error.get(name, "")) for name in ("message", "stdout", "stderr")
    ).lower()
    if "generation" in failed and "cycle" in generation_text:
        return (
            "canonical_tablebase_recurrence_blocked",
            False,
            "native recurrence cycle",
        )
    if failed & {"generation", "scalability", "projection_18"}:
        return (
            "canonical_tablebase_budget_exceeded",
            False,
            f"resource gate failed: {sorted(failed & {'generation', 'scalability', 'projection_18'})[0]}",
        )
    full_gates = {
        "transition_parity",
        "root_action_oracle",
        "store_offset_invariance",
        "sample_oracle",
        "determinism_8",
        "determinism_12",
        "lookup_benchmark",
        "projection_18",
    }
    if full_validation and all(statuses.get(name) == "passed" for name in full_gates):
        if (
            statuses.get("scalability") == "passed"
            and statuses.get("lookup_benchmark") == "passed"
            and statuses.get("projection_18") == "passed"
        ):
            return (
                "canonical_tablebase_feasible",
                True,
                "all correctness and budget gates passed",
            )
        return (
            "canonical_tablebase_correct_but_not_scalable",
            False,
            "correctness passed but scalability was not completed",
        )
    return (
        "canonical_tablebase_validation_incomplete",
        False,
        "required validation gates were not completed",
    )


def reportable_projection(result: dict[str, object]) -> dict[str, object]:
    gates = result["gates"]
    assert isinstance(gates, dict)
    failed = sorted(name for name, gate in gates.items() if gate["status"] == "failed")
    skipped = sorted(
        name for name, gate in gates.items() if gate["status"] == "skipped"
    )
    generation = result.get("generation")
    return {
        "classification": result["classification"],
        "validation_complete": result["validation_complete"],
        "failed_gates": failed,
        "skipped_gates": skipped,
        "generation": generation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="run exhaustive semantic gates and 9-12 sample oracle",
    )
    parser.add_argument(
        "--run-scalability",
        action="store_true",
        help="generate independent scalability measurements at tiers 8, 10, 12, and 14",
    )
    parser.add_argument(
        "--generator-memory-limit-kib", type=int, default=GENERATOR_MEMORY_LIMIT_KIB
    )
    parser.add_argument(
        "--generator-timeout-seconds", type=int, default=GENERATOR_TIMEOUT_SECONDS
    )
    args = parser.parse_args()
    if not 0 <= args.tier <= 20:
        parser.error("tier must be 0..20")
    if args.generator_memory_limit_kib <= 0 or args.generator_timeout_seconds <= 0:
        parser.error("generator limits must be positive")
    result: dict[str, object] = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "cpu_count": os.cpu_count(),
        },
        "generator_limits": {
            "address_space_kib": args.generator_memory_limit_kib,
            "cpu_seconds": args.generator_timeout_seconds,
            "wall_seconds": args.generator_timeout_seconds,
        },
        "gates": {},
    }
    gates = result["gates"]
    assert isinstance(gates, dict)
    generator_limits = {
        "memory_limit_kib": args.generator_memory_limit_kib,
        "timeout_seconds": args.generator_timeout_seconds,
    }
    run_gate(gates, "rank_unrank", rank_gate)
    binary_data = run_gate(gates, "build", build)
    if binary_data is None:
        for name in ("generation", "portable_format", "cited_positions"):
            skip_gate(gates, name, "native binary was not built")
    else:
        binary = binary_data
        try:
            with tempfile.TemporaryDirectory(
                prefix="kalah_v1_tablebase_", dir="/tmp"
            ) as temporary:
                directory = Path(temporary)
                tablebase = directory / f"kalah_v1_{args.tier}.kvtb"
                generation = run_gate(
                    gates,
                    "generation",
                    generate,
                    binary,
                    args.tier,
                    tablebase,
                    **generator_limits,
                )
                result["generation"] = gates["generation"]
                if generation is None:
                    for name in ("portable_format", "cited_positions"):
                        skip_gate(gates, name, "tablebase generation failed")
                else:
                    run_gate(
                        gates,
                        "portable_format",
                        portable_format_gate,
                        binary,
                        tablebase,
                        directory,
                    )
                    run_gate(gates, "cited_positions", cited_gate, binary, tablebase)
                if args.full_validation and args.tier < 12:
                    gates["full_validation_configuration"] = {
                        "status": "failed",
                        "error": {
                            "type": "ValueError",
                            "message": "full validation requires --tier >= 12",
                        },
                    }
                elif args.full_validation:
                    run_gate(gates, "transition_parity", transition_gate, binary, 8)
                    run_gate(
                        gates,
                        "root_action_oracle",
                        exact_oracle_gate,
                        binary,
                        tablebase,
                        8,
                    )
                    run_gate(
                        gates,
                        "store_offset_invariance",
                        store_offset_gate,
                        binary,
                        tablebase,
                        8,
                    )
                    run_gate(gates, "sample_oracle", sample_gate, binary, tablebase)
                    run_gate(
                        gates,
                        "determinism_8",
                        determinism_gate,
                        binary,
                        8,
                        directory,
                        **generator_limits,
                    )
                    run_gate(
                        gates,
                        "determinism_12",
                        determinism_gate,
                        binary,
                        12,
                        directory,
                        **generator_limits,
                    )
                correctness_failed = any(
                    gates.get(name, {}).get("status") == "failed"
                    for name in (
                        "rank_unrank",
                        "portable_format",
                        "cited_positions",
                        "transition_parity",
                        "root_action_oracle",
                        "store_offset_invariance",
                        "sample_oracle",
                        "determinism_8",
                        "determinism_12",
                    )
                )
                if args.run_scalability and not correctness_failed:
                    scalability = run_gate(
                        gates,
                        "scalability",
                        scalability_gate,
                        binary,
                        directory,
                        **generator_limits,
                    )
                    if scalability is not None:
                        run_gate(
                            gates,
                            "lookup_benchmark",
                            lookup_benchmark_gate,
                            binary,
                            directory / "scalability-14.kvtb",
                        )
                        run_gate(gates, "projection_18", projection_gate, scalability)
                elif args.run_scalability:
                    for name in ("scalability", "lookup_benchmark", "projection_18"):
                        skip_gate(gates, name, "correctness gate failed")
        except Exception as error:
            gates["runner"] = {"status": "failed", "error": _error_details(error)}
    if args.full_validation:
        for name in (
            "transition_parity",
            "root_action_oracle",
            "store_offset_invariance",
            "sample_oracle",
            "determinism_8",
            "determinism_12",
            "lookup_benchmark",
            "projection_18",
        ):
            if name not in gates:
                skip_gate(gates, name, "full validation did not reach this gate")
    classification, validation_complete, reason = classify(gates, args.full_validation)
    result["classification"] = classification
    result["validation_complete"] = validation_complete
    result["classification_reason"] = reason
    result["reportable"] = reportable_projection(result)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return (
        1
        if classification
        in {
            "canonical_tablebase_incorrect",
            "canonical_tablebase_recurrence_blocked",
            "canonical_tablebase_budget_exceeded",
        }
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
