"""Regression coverage for the isolated native canonical KVTB adapter."""

import hashlib
from math import comb
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ml.alphazero_lite.exact_kalah_solver import ExactState
from ml.alphazero_lite.native_mtdf_probe import NativeProbe, payload
from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import unrank
from ml.alphazero_lite.run_kalah_v1_tier18_experiment import KvtbReader


def _fixture() -> bytearray:
    """Make a valid, tiny schema-1 file without depending on a generator."""
    top = 2
    states = sum(2 * comb(tier + 11, 11) for tier in range(top + 1))
    header_size = 80 + 16 * (top + 1)
    header = bytearray(b"KVTB1\x01\x00kalah_v1\x01\x01\x80")
    header.extend((top).to_bytes(1, "little"))
    header.extend((2).to_bytes(4, "little"))
    header.extend((20).to_bytes(1, "little"))
    header.extend(states.to_bytes(8, "little") * 2)
    header.extend((header_size + states).to_bytes(8, "little"))
    offset = 0
    for tier in range(top + 1):
        count = 2 * comb(tier + 11, 11)
        header.extend(count.to_bytes(8, "little"))
        header.extend(offset.to_bytes(8, "little"))
        offset += count
    body = bytes(states)
    header.extend(hashlib.sha256(body).digest())
    assert len(header) == header_size
    return header + body


class CanonicalKvtbAdapterTest(unittest.TestCase):
    def _native(self) -> Path:
        executable = os.environ.get("NATIVE_MTDF_PROBE")
        if not executable:
            self.skipTest("native probe is built by the native CI job")
        return Path(executable)

    def _loads(self, data: bytes | bytearray) -> bool:
        executable = self._native()
        with tempfile.NamedTemporaryFile() as file:
            file.write(data)
            file.flush()
            result = subprocess.run(
                [str(executable)], env=os.environ | {"NATIVE_CANONICAL_KVTB": file.name}
            )
        return result.returncode == 0

    def test_rejects_portable_format_malformed_files(self):
        base = _fixture()
        cases = {
            "empty": bytearray(),
            "magic": bytearray(base),
            "schema": bytearray(base),
            "game_id": bytearray(base),
            "player_encoding": bytearray(base),
            "byte_order": bytearray(base),
            "unknown_value": bytearray(base),
            "tier": bytearray(base),
            "generator_revision": bytearray(base),
            "max_tier": bytearray(base),
            "state_count": bytearray(base),
            "payload_length": bytearray(base),
            "file_length": bytearray(base),
            "tier_state_count": bytearray(base),
            "tier_offset": bytearray(base),
            "checksum": bytearray(base),
            "payload": bytearray(base),
            "truncated": bytearray(base[:-1]),
            "trailing_data": bytearray(base + b"x"),
        }
        cases["magic"][0] ^= 1
        cases["schema"][5] = 2
        cases["game_id"][7] ^= 1
        cases["player_encoding"][15] = 2
        cases["byte_order"][16] = 2
        cases["unknown_value"][17] = 0
        cases["tier"][18] = 21
        cases["generator_revision"][19] ^= 1
        cases["max_tier"][23] = 19
        cases["state_count"][24] ^= 1
        cases["payload_length"][32] ^= 1
        cases["file_length"][40] ^= 1
        cases["tier_state_count"][48] ^= 1
        cases["tier_offset"][56 + 16] ^= 1
        cases["checksum"][127] ^= 1
        cases["payload"][-1] ^= 1
        self.assertTrue(self._loads(base))
        for name, malformed in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._loads(malformed))

    def test_every_tier_offset_and_orientation_match_canonical_reader(self):
        artifact = os.environ.get("NATIVE_CANONICAL_KVTB")
        if not artifact:
            self.skipTest("validated tier-18 artifact is not available")
        reader = KvtbReader(Path(artifact))
        probe = NativeProbe(self._native())
        try:
            # A nonzero rank makes the old +8 metadata expression observable.
            for tier in range(19):
                pits = (
                    unrank(tier, min(1, comb(tier + 11, 11) - 1))
                    if tier < 2
                    else (1, 0, 0, 0, 0, 0, tier - 1, 0, 0, 0, 0, 0)
                )
                for player in (0, 1):
                    state = ExactState(pits, (23, 17), player)
                    result = probe.request(payload(state, "diagnose"))
                    with self.subTest(tier=tier, player=player):
                        self.assertEqual(tier, result["active_stones"])
                        self.assertEqual(reader.offsets[tier], result["offset"])
                        self.assertEqual(
                            reader.value(pits, player), result["raw_value"]
                        )
                        self.assertEqual(6, result["store_margin"])
                        global_margin = 6 + result["raw_value"]
                        self.assertEqual(
                            -global_margin if player else global_margin,
                            result["upstream_value"],
                        )
                        self.assertEqual(global_margin, result["player_zero_value"])
                    if state.legal_moves():
                        label = probe.request(payload(state, "label"))
                        expected_actions = {}
                        for action in state.legal_moves():
                            child = state.play(action)
                            expected_actions[action] = (
                                child.settled_margin()
                                if child.is_terminal()
                                else child.stores[0]
                                - child.stores[1]
                                + reader.value(child.pits, child.current_player)
                            )
                        with self.subTest(tier=tier, player=player, actions=True):
                            self.assertEqual(
                                expected_actions,
                                {
                                    int(key): value
                                    for key, value in label["action_values"].items()
                                },
                            )
        finally:
            probe.close()
