import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.alphazero_lite.exact_kalah_solver import ExactKalahSolver, ExactState
from ml.alphazero_lite.kalah_rules import KalahGame


class ExactKalahSolverTest(unittest.TestCase):
    def test_independent_transition_matches_capture_and_sweep(self):
        state = ExactState.from_game_state(
            {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [0, 0, 0, 0, 3, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        expected = KalahGame.from_state(state.to_game_state())
        expected.move(expected.pit_index(0))

        self.assertEqual(expected.to_state(), state.play(0).to_game_state())
        self.assertEqual(4, state.play(0).settled_margin())

    def test_solver_matches_bruteforce_on_tiny_position(self):
        state = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
        solver = ExactKalahSolver(cache_enabled=False)

        self.assertEqual(_bruteforce(state), solver.solve(state))
        self.assertEqual(
            {move: _bruteforce(state.play(move)) for move in state.legal_moves()},
            solver.action_margins(state),
        )

    def test_exact_result_is_invariant_to_order_and_tt_size(self):
        state = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
        ascending = ExactKalahSolver(
            cache_enabled=False, tt_size=2, move_order="ascending"
        )
        descending = ExactKalahSolver(
            cache_enabled=False, tt_size=100, move_order="descending"
        )

        self.assertEqual(ascending.solve(state), descending.solve(state))
        self.assertEqual(ascending.solve(state), ascending.solve(state))

    def test_exact_values_persist_across_solver_instances(self):
        state = ExactState((0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0), (20, 20), 0)
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "exact.sqlite3"
            first = ExactKalahSolver(cache_path=cache_path)
            expected = first.solve(state)
            first.close()
            second = ExactKalahSolver(cache_path=cache_path)
            self.assertEqual(expected, second.solve(state))
            self.assertGreater(second.cache_hits, 0)
            second.close()


def _bruteforce(state: ExactState) -> int:
    if state.is_terminal():
        return state.settled_margin()
    values = [_bruteforce(state.play(move)) for move in state.legal_moves()]
    return max(values) if state.current_player == 0 else min(values)
