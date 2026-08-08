import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.runtime_profiles import runtime_profile_definition


def load_gate_module():
    path = Path(__file__).resolve().parents[2] / "script/ai/seat_aware_promotion_gate"
    loader = importlib.machinery.SourceFileLoader("seat_aware_runtime_gate", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class SeatAwareRuntimeProfileGateTests(unittest.TestCase):
    @staticmethod
    def profile(name: str, bias: float) -> dict:
        return runtime_profile_definition(
            name=name,
            default_tactical_root_bias=bias,
            tactical_root_bias_overrides={},
            default_c_puct=1.25,
            c_puct_overrides={"768:768": 0.90},
        )

    def test_gate_accepts_explicit_distinct_side_profiles(self):
        gate = load_gate_module()
        candidate = self.profile("candidate", 0.10)
        current = self.profile("current", 0.00)
        arguments = [
            "seat_aware_promotion_gate",
            "--candidate-path",
            "model-artifact/current",
            "--out",
            "/tmp/runtime-profile-gate.json",
            "--candidate-runtime-profile-json",
            json.dumps(candidate),
            "--current-runtime-profile-json",
            json.dumps(current),
        ]
        with patch.object(sys, "argv", arguments):
            args = gate.parse_args()
        self.assertNotEqual(
            gate.parse_runtime_profile_json(args.candidate_runtime_profile_json)[
                "hash"
            ],
            gate.parse_runtime_profile_json(args.current_runtime_profile_json)["hash"],
        )

    def test_gate_rejects_equal_profiles_without_explicit_null_mode(self):
        gate = load_gate_module()
        profile = self.profile("same", 0.00)
        arguments = [
            "seat_aware_promotion_gate",
            "--candidate-path",
            "model-artifact/current",
            "--out",
            "/tmp/runtime-profile-gate.json",
            "--candidate-runtime-profile-json",
            json.dumps(profile),
            "--current-runtime-profile-json",
            json.dumps(profile),
        ]
        with patch.object(sys, "argv", arguments):
            args = gate.parse_args()
        with self.assertRaisesRegex(ValueError, "identical runtime-profile hashes"):
            gate.run_seat_aware_evaluation(args)


if __name__ == "__main__":
    unittest.main()
