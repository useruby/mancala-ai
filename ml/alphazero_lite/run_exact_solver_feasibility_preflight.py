"""Run the exact-teacher feasibility preflight without production integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from collections.abc import Callable
from typing import Any

from ml.alphazero_lite.exact_kalah_solver import (
    ExactKalahSolver,
    ExactState,
    SearchTimeout,
)
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state


BUCKETS = ((17, 24), (25, 32), (33, 40))
STATES_PER_BUCKET = 32
DEFAULT_SEED = 271


def standard_initial_state() -> ExactState:
    """Return the standard six-pit, four-stone Kalah opening position."""
    return ExactState((4,) * 12, (0, 0), 0)


def run_rules_parity(
    sample_count: int = 10_000, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Compare all golden and deterministic reachable transitions to KalahGame."""
    fixture = Path("test/fixtures/ai/kalah_rule_vectors.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    checked_moves = 0
    event_counts = {"extra_turns": 0, "captures": 0, "terminal_sweeps": 0}
    for vector in payload["vectors"]:
        expected = KalahGame.from_state(vector["initial_state"])
        actual = ExactState.from_game_state(vector["initial_state"])
        for step in vector["steps"]:
            move = int(step["relative_move"])
            _assert_transition_parity(actual, expected, move, f"golden:{vector['id']}")
            actual = actual.play(move)
            expected.move(expected.pit_index(move))
            checked_moves += 1

    rng = random.Random(seed)
    checked_states = 0
    while checked_states < sample_count:
        actual = standard_initial_state()
        expected = KalahGame.from_state(actual.to_game_state())
        while not actual.is_terminal() and checked_states < sample_count:
            moves = actual.legal_moves()
            for move in moves:
                consequence = move_consequence_for_state(actual.to_game_state(), move)
                _assert_transition_parity(
                    actual, expected, move, f"reachable:{checked_states}:{move}"
                )
                event_counts["extra_turns"] += int(consequence["gives_extra_turn"])
                event_counts["captures"] += int(consequence["produces_capture"])
                event_counts["terminal_sweeps"] += int(
                    consequence["game_over_after_move"]
                )
                checked_moves += 1
            move = rng.choice(moves)
            actual = actual.play(move)
            expected.move(expected.pit_index(move))
            checked_states += 1
    return {
        "passed": True,
        "golden_vectors": len(payload["vectors"]),
        "reachable_states": checked_states,
        "legal_moves_checked": checked_moves,
        "events_observed": event_counts,
    }


def _assert_transition_parity(
    actual: ExactState, expected: KalahGame, move: int, context: str
) -> None:
    child = actual.play(move)
    reference = expected.clone()
    if not reference.move(reference.pit_index(move)):
        raise AssertionError(f"reference rejected legal move at {context}")
    if child.to_game_state() != reference.to_state():
        raise AssertionError(f"state mismatch at {context}")
    if child.is_terminal() != reference.over():
        raise AssertionError(f"terminal mismatch at {context}")
    winner = _winner(child)
    if winner != reference.winner:
        raise AssertionError(
            f"winner mismatch at {context}: {winner} != {reference.winner}"
        )
    if child.settled_margin() != _settled_margin(reference):
        raise AssertionError(f"final-margin mismatch at {context}")


def run_exactness_validation(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Cross-check tiny positions against exhaustive minimax under solver variants."""
    states = _tiny_reachable_states(seed)
    checked = 0
    for state in states:
        expected = _bruteforce(state)
        results = []
        for order, tt_size, cache_enabled in (
            ("ascending", 2, False),
            ("descending", 17, False),
            ("ascending", 1_000, True),
        ):
            solver = ExactKalahSolver(
                tt_size=tt_size, move_order=order, cache_enabled=cache_enabled
            )
            try:
                results.extend((solver.solve(state), solver.solve(state)))
            finally:
                solver.close()
        if any(value != expected for value in results):
            raise AssertionError(f"exactness mismatch for {state}")
        checked += 1
    return {"passed": True, "tiny_positions": checked, "executions": checked * 6}


def generate_feasibility_corpus(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Generate deterministic, newly sampled states marked ineligible for training."""
    rng = random.Random(seed)
    selected: dict[tuple[int, int], list[ExactState]] = {
        bucket: [] for bucket in BUCKETS
    }
    seen: set[bytes] = set()
    attempts = 0
    while any(len(states) < STATES_PER_BUCKET for states in selected.values()):
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not generate all requested stone-count buckets")
        state = standard_initial_state()
        while not state.is_terminal():
            stones = sum(state.pits)
            for bucket, states in selected.items():
                if bucket[0] <= stones <= bucket[1] and len(states) < STATES_PER_BUCKET:
                    if state.key() not in seen:
                        seen.add(state.key())
                        states.append(state)
                    break
            state = state.play(rng.choice(state.legal_moves()))

    rows = []
    for bucket in BUCKETS:
        for index, state in enumerate(selected[bucket]):
            rows.append(
                {
                    "id": f"exact-preflight-{bucket[0]}-{bucket[1]}-{index:02d}",
                    "state": state.to_game_state(),
                    "stones_remaining": sum(state.pits),
                    "training_eligible": False,
                    "provenance": "fresh deterministic exact-solver feasibility corpus",
                }
            )
    return rows


def run_feasibility(
    corpus: list[dict[str, Any]],
    cache_path: Path,
    time_limit_seconds: float,
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Solve every corpus state and retain exact per-action final margins."""
    solver = ExactKalahSolver(cache_path=cache_path)
    rows = []
    try:
        for source in corpus:
            state = ExactState.from_game_state(source["state"])
            started = time.monotonic()
            try:
                margins = solver.action_margins(
                    state, time_limit_seconds=time_limit_seconds
                )
                exact = True
                first_nodes = solver.nodes
                first_cache_hits = solver.cache_hits
                repeated_margins = solver.action_margins(
                    state, time_limit_seconds=time_limit_seconds
                )
                reproducible = margins == repeated_margins
                if not reproducible:
                    raise AssertionError(
                        f"non-reproducible action margins for {source['id']}"
                    )
            except SearchTimeout:
                margins = {}
                exact = False
                first_nodes = solver.nodes
                first_cache_hits = solver.cache_hits
                reproducible = False
            runtime = time.monotonic() - started
            root_value = (
                (
                    max(margins.values())
                    if state.current_player == 0
                    else min(margins.values())
                )
                if margins
                else None
            )
            optimal_actions = (
                []
                if root_value is None
                else sorted(
                    move for move, value in margins.items() if value == root_value
                )
            )
            rows.append(
                source
                | {
                    "exact": exact,
                    "runtime_seconds": runtime,
                    "node_count": first_nodes,
                    "cache_hits": first_cache_hits,
                    "cache_used": first_cache_hits > 0,
                    "reproducible": reproducible,
                    "final_margin": root_value,
                    "optimal_actions": optimal_actions,
                    "action_final_margins": margins,
                }
            )
            if on_row is not None:
                on_row(rows)
    finally:
        solver.close()
    return rows


def qualification(
    rows: list[dict[str, Any]], parity: dict[str, Any], exactness: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the predeclared feasibility thresholds."""
    by_bucket = {}
    for low, high in BUCKETS:
        bucket_rows = [row for row in rows if low <= row["stones_remaining"] <= high]
        solved = [row for row in bucket_rows if row["exact"]]
        by_bucket[f"{low}-{high}"] = {
            "solved": len(solved),
            "total": len(bucket_rows),
            "rate": len(solved) / len(bucket_rows) if bucket_rows else 0.0,
        }
    solved_rows = [row for row in rows if row["exact"]]
    mean_runtime = sum(row["runtime_seconds"] for row in solved_rows) / max(
        1, len(solved_rows)
    )
    projected = int(24 * 60 * 60 / mean_runtime) if solved_rows else 0
    reproducible = all(
        row["reproducible"] and row["optimal_actions"] for row in solved_rows
    )
    passed = (
        parity["passed"]
        and exactness["passed"]
        and by_bucket["17-24"]["rate"] == 1.0
        and by_bucket["25-32"]["rate"] >= 0.75
        and by_bucket["33-40"]["rate"] >= 0.25
        and reproducible
        and projected >= 50_000
    )
    return {
        "passed": passed,
        "bucket_coverage": by_bucket,
        "reproducible_optimal_action_sets": reproducible,
        "mean_exact_runtime_seconds": mean_runtime,
        "projected_exact_labels_24_cpu_hours": projected,
        "dataset_scale_target": 50_000,
    }


def _tiny_reachable_states(seed: int) -> list[ExactState]:
    rng = random.Random(seed)
    states = []
    while len(states) < 12:
        state = standard_initial_state()
        while not state.is_terminal():
            if 4 <= sum(state.pits) <= 8:
                states.append(state)
                break
            state = state.play(rng.choice(state.legal_moves()))
    return states


def _bruteforce(state: ExactState) -> int:
    if state.is_terminal():
        return state.settled_margin()
    values = [_bruteforce(state.play(move)) for move in state.legal_moves()]
    return max(values) if state.current_player == 0 else min(values)


def _settled_margin(game: KalahGame) -> int:
    scores = game.captured_seeds.copy()
    for player in (0, 1):
        offset = player * 6
        scores[player] += sum(game.pits[offset : offset + 6])
    return scores[0] - scores[1]


def _winner(state: ExactState) -> int | None:
    if not state.is_terminal():
        return None
    margin = state.settled_margin()
    return 0 if margin > 0 else 1 if margin < 0 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--parity-samples", type=int, default=10_000)
    parser.add_argument("--bucket", choices=[f"{low}-{high}" for low, high in BUCKETS])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parity = run_rules_parity(args.parity_samples, args.seed)
    exactness = run_exactness_validation(args.seed)
    corpus = generate_feasibility_corpus(args.seed)
    if args.bucket is not None:
        low, high = (int(value) for value in args.bucket.split("-"))
        corpus = [row for row in corpus if low <= row["stones_remaining"] <= high]

    def write_report(rows: list[dict[str, Any]], complete: bool) -> None:
        report = {
            "schema": "exact_kalah_solver_feasibility_v1",
            "seed": args.seed,
            "complete": complete,
            "rules_parity": parity,
            "exactness_validation": exactness,
            "corpus": rows,
            "qualification": qualification(rows, parity, exactness)
            if complete
            else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    rows = run_feasibility(
        corpus,
        args.cache,
        args.time_limit_seconds,
        on_row=lambda completed: write_report(completed, False),
    )
    write_report(rows, True)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    return 0 if report["qualification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
