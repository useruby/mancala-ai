import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase


class FakeInput:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        pass


class FakeOutput:
    def readline(self):
        return '{"actions":{"1":-6,"4":6,"5":0}}\n'


class FakeProcess:
    def __init__(self):
        self.stdin = FakeInput()
        self.stdout = FakeOutput()
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.wait_timeout = timeout


class NativeExactRootTablebaseTest(unittest.TestCase):
    def test_converts_native_player_zero_pit_margins_to_root_final_margins(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 6, 0, 0, 1, 2],
                "opponent_pits": [0, 0, 3, 0, 0, 4],
                "player_store": 13,
                "opponent_store": 19,
                "current_player": 0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            tablebase = Path(directory) / "tier16.kvtb"
            binary.touch()
            tablebase.touch()
            process = FakeProcess()
            with patch(
                "ml.alphazero_lite.native_exact_root_tablebase.subprocess.Popen",
                return_value=process,
            ):
                oracle = NativeExactRootTablebase(binary, tablebase)
                self.assertEqual(
                    {1: -12, 4: 0, 5: -6}, oracle.root_action_margins(game, 0)
                )
                oracle.close()

        self.assertEqual(1, oracle.calls)
        self.assertTrue(process.terminated)
        self.assertIn('"player":0', process.stdin.writes[0])

    def test_negates_player_zero_margin_for_player_one_root(self):
        game = KalahGame.from_state(
            {
                "player_pits": [0, 6, 0, 0, 1, 2],
                "opponent_pits": [0, 0, 3, 0, 0, 4],
                "player_store": 13,
                "opponent_store": 19,
                "current_player": 1,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            tablebase = Path(directory) / "tier16.kvtb"
            binary.touch()
            tablebase.touch()
            with patch(
                "ml.alphazero_lite.native_exact_root_tablebase.subprocess.Popen",
                return_value=FakeProcess(),
            ):
                with NativeExactRootTablebase(binary, tablebase) as oracle:
                    self.assertEqual(
                        {1: 12, 4: 0, 5: 6}, oracle.root_action_margins(game, 1)
                    )
