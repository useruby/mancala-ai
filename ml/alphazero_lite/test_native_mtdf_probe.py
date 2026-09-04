import os
from pathlib import Path
import unittest

from ml.alphazero_lite.exact_kalah_solver import ExactState
from ml.alphazero_lite.native_mtdf_probe import NativeProbe, payload


class NativeMtdfProbeTest(unittest.TestCase):
    def test_cited_tiny_mismatch_is_reproducible(self):
        executable = os.environ.get("NATIVE_MTDF_PROBE")
        if not executable:
            self.skipTest("native probe is built by the native CI job")
        state = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
        probe = NativeProbe(Path(executable))
        try:
            result = probe.request(payload(state, "label"))
        finally:
            probe.close()
        self.assertEqual({"1": -4, "2": 2}, result["action_values"])
