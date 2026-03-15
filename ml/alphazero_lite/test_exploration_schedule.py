import unittest

from ml.alphazero_lite import self_play


class ExplorationScheduleTest(unittest.TestCase):
    def test_schedule_is_noop_in_none_mode(self):
        params = self_play.schedule_exploration_params(
            mode="none",
            iteration=3,
            total_iterations=5,
            temperature=1.0,
            temperature_late=0.1,
            temperature_threshold=10,
            dirichlet_epsilon=0.25,
        )

        self.assertEqual(1.0, params["temperature"])
        self.assertEqual(0.1, params["temperature_late"])
        self.assertEqual(10, params["temperature_threshold"])
        self.assertEqual(0.25, params["dirichlet_epsilon"])

    def test_linear_schedule_decays_exploration_with_progress(self):
        early = self_play.schedule_exploration_params(
            mode="linear",
            iteration=1,
            total_iterations=5,
            temperature=1.0,
            temperature_late=0.2,
            temperature_threshold=12,
            dirichlet_epsilon=0.25,
        )
        late = self_play.schedule_exploration_params(
            mode="linear",
            iteration=5,
            total_iterations=5,
            temperature=1.0,
            temperature_late=0.2,
            temperature_threshold=12,
            dirichlet_epsilon=0.25,
        )

        self.assertLess(late["temperature"], early["temperature"])
        self.assertLess(late["temperature_late"], early["temperature_late"])
        self.assertLess(late["dirichlet_epsilon"], early["dirichlet_epsilon"])
        self.assertLess(late["temperature_threshold"], early["temperature_threshold"])


if __name__ == "__main__":
    unittest.main()
