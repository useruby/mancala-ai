import os
from pathlib import Path
import unittest

from ml.alphazero_lite.exact_kalah_solver import ExactState
from ml.alphazero_lite.native_mtdf_probe import (
    NativeProbe,
    payload,
    tiny_exactness_report,
    transition_report,
)


class NativeMtdfProbeTest(unittest.TestCase):
    def test_cited_tiny_position_matches_oracle(self):
        executable = os.environ.get("NATIVE_MTDF_PROBE")
        if not executable:
            self.skipTest("native probe is built by the native CI job")
        state = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
        probe = NativeProbe(Path(executable))
        try:
            result = probe.request(payload(state, "label"))
        finally:
            probe.close()
        self.assertEqual({"1": -4, "2": 0}, result["action_values"])

    def test_tiny_exactness_gate(self):
        executable = os.environ.get("NATIVE_MTDF_PROBE")
        if not executable:
            self.skipTest("native probe is built by the native CI job")
        report = tiny_exactness_report(Path(executable))
        self.assertTrue(report["passed"])
        self.assertEqual(12, report["tiny_positions"])

    def test_full_native_transition_gate(self):
        executable = os.environ.get("NATIVE_MTDF_PROBE")
        if not executable:
            self.skipTest("native probe is built by the native CI job")
        report = transition_report(Path(executable))
        self.assertEqual(10_000, report["native_reachable_states"])
        self.assertGreater(report["native_transitions_checked"], 10_000)
