"""Reproducible driver for the pinned native MTD(f) compatibility probe."""

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
from typing import Any

from ml.alphazero_lite.exact_kalah_solver import ExactKalahSolver, ExactState
from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
    DEFAULT_SEED,
    _tiny_reachable_states,
    run_rules_parity,
)


class NativeProbe:
    def __init__(self, executable: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self) -> None:
        self.process.terminate()
        self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def payload(
    state: ExactState, operation: str, action: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "operation": operation,
        "pits": list(state.pits),
        "stores": list(state.stores),
        "player": state.current_player,
    }
    if action is not None:
        result["action"] = action
    return result


def transition_report(executable: Path, sample_count: int = 10_000) -> dict[str, Any]:
    # Keep the independently implemented Python parity gate, then exercise the
    # native process for the same deterministic reachable-state construction.
    baseline = run_rules_parity(sample_count, DEFAULT_SEED)
    probe = NativeProbe(executable)
    checked = 0
    states = 0
    try:
        fixture = json.loads(
            Path("test/fixtures/ai/kalah_rule_vectors.json").read_text(encoding="utf-8")
        )
        for vector in fixture["vectors"]:
            state = ExactState.from_game_state(vector["initial_state"])
            for step in vector["steps"]:
                _assert_native_transition(probe, state, int(step["relative_move"]))
                checked += 1
                state = state.play(int(step["relative_move"]))

        rng = random.Random(DEFAULT_SEED)
        while states < sample_count:
            state = ExactState((4,) * 12, (0, 0), 0)
            while not state.is_terminal() and states < sample_count:
                for action in state.legal_moves():
                    _assert_native_transition(probe, state, action)
                    checked += 1
                state = state.play(rng.choice(state.legal_moves()))
                states += 1
    finally:
        probe.close()
    return baseline | {
        "native_reachable_states": states,
        "native_transitions_checked": checked,
    }


def _assert_native_transition(
    probe: NativeProbe, state: ExactState, action: int
) -> None:
    got = probe.request(payload(state, "apply", action))
    child = state.play(action)
    expected = {
        "pits": list(child.pits),
        "stores": list(child.stores),
        "player": child.current_player,
        "extra_turn": child.current_player == state.current_player
        and not child.is_terminal(),
        "terminal": child.is_terminal(),
        "final_margin": child.settled_margin(),
    }
    if any(got[key] != value for key, value in expected.items()):
        raise AssertionError(
            {"state": state, "action": action, "got": got, "expected": expected}
        )


def tiny_exactness_report(executable: Path) -> dict[str, Any]:
    cases = _tiny_reachable_states(DEFAULT_SEED)
    cited = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
    cases[0] = cited
    rows = []
    probe = NativeProbe(executable)
    try:
        for state in cases:
            oracle = ExactKalahSolver(cache_enabled=False)
            try:
                expected = oracle.action_margins(state)
            finally:
                oracle.close()
            first = probe.request(payload(state, "label"))
            second = probe.request(payload(state, "label"))
            actual = {int(key): value for key, value in first["action_values"].items()}
            root = (
                max(expected.values())
                if state.current_player == 0
                else min(expected.values())
            )
            rows.append(
                {
                    "input": payload(state, "label"),
                    "native": first,
                    "expected_action_values": expected,
                    "expected_root_value": root,
                    "passed": actual == expected
                    and first["exact_value"] == root
                    and first["action_values"] == second["action_values"]
                    and first["exact_value"] == second["exact_value"]
                    and first["optimal_actions"] == second["optimal_actions"],
                }
            )
    finally:
        probe.close()
    return {
        "passed": all(row["passed"] for row in rows),
        "tiny_positions": len(rows),
        "rows": rows,
    }


def write_correctness_results(executable: Path, output: Path) -> bool:
    transition = transition_report(executable)
    exactness = tiny_exactness_report(executable)
    report = {
        "schema": "native_mtdf_correctness_v1",
        "transition": transition,
        "tiny_exactness": exactness,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bool(exactness["passed"])
