"""Reproducible driver for the pinned native MTD(f) compatibility probe."""

from __future__ import annotations

import json
from pathlib import Path
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
    # The existing independent parity generator supplies golden and reachable states.
    baseline = run_rules_parity(sample_count, DEFAULT_SEED)
    probe = NativeProbe(executable)
    checked = 0
    try:
        for state in _tiny_reachable_states(DEFAULT_SEED):
            for action in state.legal_moves():
                got = probe.request(payload(state, "apply", action))
                child = state.play(action)
                expected = {
                    "pits": list(child.pits),
                    "stores": list(child.stores),
                    "player": child.current_player,
                    "terminal": child.is_terminal(),
                    "final_margin": child.settled_margin(),
                }
                if any(got[key] != value for key, value in expected.items()):
                    raise AssertionError(
                        {
                            "state": state,
                            "action": action,
                            "got": got,
                            "expected": expected,
                        }
                    )
                checked += 1
    finally:
        probe.close()
    return baseline | {"native_transitions_checked": checked}


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
                    and first == second,
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
