import json
import os
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seat_aware_arena import (
    BUDGET_PAIR_LABELS,
    build_candidate_ranking,
    build_seat_aware_report,
    classify_candidate,
    compute_seat_split_metrics,
    parse_game_jsonl,
    sha256_file,
)


def _make_game_entry(
    game_index=0,
    challenger_player=0,
    first_move_challenger=1,
    first_move_current=2,
    margin=5,
    game_length=30,
    winner="challenger",
    trajectory="1,2,3,4,5",
):
    return {
        "game_index": game_index,
        "challenger_player": challenger_player,
        "first_move_challenger": first_move_challenger,
        "first_move_current": first_move_current,
        "margin": margin,
        "game_length": game_length,
        "winner": winner,
        "trajectory": trajectory,
    }


class ParseGameJsonlTest(unittest.TestCase):
    def test_parses_valid_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_game_entry(game_index=0)) + "\n")
            f.write(json.dumps(_make_game_entry(game_index=1)) + "\n")
            f.flush()
            path = f.name

        try:
            entries = parse_game_jsonl(path)
            self.assertEqual(2, len(entries))
            self.assertEqual(0, entries[0]["game_index"])
            self.assertEqual(1, entries[1]["game_index"])
        finally:
            os.unlink(path)

    def test_ignores_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_game_entry(game_index=0)) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_game_entry(game_index=1)) + "\n")
            f.flush()
            path = f.name

        try:
            entries = parse_game_jsonl(path)
            self.assertEqual(2, len(entries))
        finally:
            os.unlink(path)


class ComputeSeatSplitMetricsTest(unittest.TestCase):
    def test_alternating_seats_produce_balanced_scores(self):
        entries = []
        for i in range(60):
            entries.append(
                _make_game_entry(
                    game_index=i * 2,
                    challenger_player=0,
                    winner="challenger",
                    margin=10,
                )
            )
        for i in range(60):
            entries.append(
                _make_game_entry(
                    game_index=i * 2 + 1,
                    challenger_player=1,
                    winner="current",
                    margin=-10,
                )
            )

        metrics = compute_seat_split_metrics(entries)

        self.assertEqual(60, metrics["challenger_starts_0"]["games"])
        self.assertEqual(60, metrics["challenger_starts_0"]["wins"])
        self.assertEqual(0, metrics["challenger_starts_0"]["losses"])
        self.assertEqual(0, metrics["challenger_starts_0"]["draws"])
        self.assertAlmostEqual(1.0, metrics["challenger_starts_0"]["score"])

        self.assertEqual(60, metrics["challenger_starts_1"]["games"])
        self.assertEqual(0, metrics["challenger_starts_1"]["wins"])
        self.assertEqual(60, metrics["challenger_starts_1"]["losses"])
        self.assertEqual(0, metrics["challenger_starts_1"]["draws"])
        self.assertAlmostEqual(0.0, metrics["challenger_starts_1"]["score"])

        self.assertAlmostEqual(0.0, metrics["disadvantaged_seat_score"])
        self.assertEqual(120, metrics["total_games"])

    def test_disadvantaged_seat_score_is_p1_score(self):
        entries = [
            _make_game_entry(
                game_index=0, challenger_player=1, winner="challenger", margin=3
            ),
            _make_game_entry(
                game_index=1, challenger_player=1, winner="challenger", margin=5
            ),
            _make_game_entry(
                game_index=2, challenger_player=1, winner="current", margin=-2
            ),
            _make_game_entry(
                game_index=3, challenger_player=1, winner="draw", margin=0
            ),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertAlmostEqual((2 + 0.5 * 1) / 4, metrics["disadvantaged_seat_score"])
        self.assertAlmostEqual(0.625, metrics["disadvantaged_seat_score"])

    def test_draws_contribute_half(self):
        entries = [
            _make_game_entry(
                game_index=0, challenger_player=1, winner="challenger", margin=1
            ),
            _make_game_entry(
                game_index=1, challenger_player=1, winner="draw", margin=0
            ),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertAlmostEqual(0.75, metrics["disadvantaged_seat_score"])

    def test_margin_by_seat(self):
        entries = [
            _make_game_entry(
                game_index=0, challenger_player=0, winner="challenger", margin=10
            ),
            _make_game_entry(
                game_index=1, challenger_player=0, winner="challenger", margin=5
            ),
            _make_game_entry(
                game_index=2, challenger_player=1, winner="current", margin=-8
            ),
            _make_game_entry(
                game_index=3, challenger_player=1, winner="current", margin=-2
            ),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertAlmostEqual(7.5, metrics["challenger_starts_0"]["margin_mean"])
        self.assertAlmostEqual(7.5, metrics["challenger_starts_0"]["margin_median"])
        self.assertAlmostEqual(-5.0, metrics["challenger_starts_1"]["margin_mean"])
        self.assertAlmostEqual(-5.0, metrics["challenger_starts_1"]["margin_median"])

    def test_game_length_by_seat(self):
        entries = [
            _make_game_entry(
                game_index=0, challenger_player=0, game_length=20, margin=1
            ),
            _make_game_entry(
                game_index=1, challenger_player=0, game_length=30, margin=1
            ),
            _make_game_entry(
                game_index=2, challenger_player=1, game_length=40, margin=-1
            ),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertAlmostEqual(25.0, metrics["challenger_starts_0"]["game_length_mean"])
        self.assertAlmostEqual(40.0, metrics["challenger_starts_1"]["game_length_mean"])

    def test_first_move_distribution(self):
        entries = [
            _make_game_entry(
                game_index=0, first_move_challenger=1, first_move_current=2, margin=1
            ),
            _make_game_entry(
                game_index=1, first_move_challenger=1, first_move_current=5, margin=1
            ),
            _make_game_entry(
                game_index=2, first_move_challenger=3, first_move_current=2, margin=1
            ),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertEqual({"1": 2, "3": 1}, metrics["first_move_challenger_dist"])
        self.assertEqual({"2": 2, "5": 1}, metrics["first_move_current_dist"])

    def test_duplicate_trajectory_detection(self):
        entries = [
            _make_game_entry(game_index=0, trajectory="1,2,3"),
            _make_game_entry(game_index=1, trajectory="1,2,3"),
            _make_game_entry(game_index=2, trajectory="4,5,6"),
            _make_game_entry(game_index=3, trajectory="1,2,3"),
        ]

        metrics = compute_seat_split_metrics(entries)
        self.assertEqual(2, metrics["unique_trajectories"])
        self.assertEqual(3, metrics["duplicate_trajectory_count"])

    def test_ci95_attached_to_seat_splits(self):
        entries = []
        for i in range(60):
            entries.append(
                _make_game_entry(
                    game_index=i, challenger_player=0, winner="challenger", margin=5
                )
            )

        metrics = compute_seat_split_metrics(entries)
        ci = metrics["challenger_starts_0"]["ci95"]
        self.assertIn("lower", ci)
        self.assertIn("upper", ci)
        self.assertGreater(ci["lower"], 0.9)
        self.assertAlmostEqual(1.0, ci["upper"])


class ClassifyCandidateTest(unittest.TestCase):
    def test_seat_artifact_only(self):
        budget_results = {
            "standard": {
                "disadvantaged_seat_score": 0.0,
                "arena_score": 0.50,
            },
            "equal_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "challenger_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "current_high_asymmetry": {
                "disadvantaged_seat_score": 0.0,
            },
        }
        result = classify_candidate(budget_results, standard_alternating_score=0.50)
        self.assertEqual("seat_artifact_only", result)

    def test_high_search_breakthrough_at_equal_high(self):
        budget_results = {
            "standard": {
                "disadvantaged_seat_score": 0.0,
                "arena_score": 0.50,
            },
            "equal_high": {
                "disadvantaged_seat_score": 0.5,
            },
            "challenger_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "current_high_asymmetry": {
                "disadvantaged_seat_score": 0.0,
            },
        }
        result = classify_candidate(budget_results, standard_alternating_score=0.50)
        self.assertEqual("high_search_breakthrough", result)

    def test_high_search_breakthrough_at_challenger_high(self):
        budget_results = {
            "standard": {
                "disadvantaged_seat_score": 0.0,
                "arena_score": 0.50,
            },
            "equal_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "challenger_high": {
                "disadvantaged_seat_score": 1.0,
            },
            "current_high_asymmetry": {
                "disadvantaged_seat_score": 0.0,
            },
        }
        result = classify_candidate(budget_results, standard_alternating_score=0.50)
        self.assertEqual("high_search_breakthrough", result)

    def test_standard_budget_breakthrough(self):
        budget_results = {
            "standard": {
                "disadvantaged_seat_score": 0.15,
                "arena_score": 0.55,
            },
            "equal_high": {
                "disadvantaged_seat_score": 0.5,
            },
            "challenger_high": {
                "disadvantaged_seat_score": 0.5,
            },
            "current_high_asymmetry": {
                "disadvantaged_seat_score": 0.0,
            },
        }
        result = classify_candidate(budget_results, standard_alternating_score=0.55)
        self.assertEqual("standard_budget_breakthrough", result)

    def test_regression_masked_by_seat(self):
        budget_results = {
            "standard": {
                "disadvantaged_seat_score": 0.0,
                "arena_score": 0.50,
            },
            "equal_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "challenger_high": {
                "disadvantaged_seat_score": 0.0,
            },
            "current_high_asymmetry": {
                "disadvantaged_seat_score": 0.0,
            },
        }
        result = classify_candidate(budget_results, standard_alternating_score=0.45)
        self.assertEqual("regression_masked_by_seat", result)


class BuildSeatAwareReportTest(unittest.TestCase):
    def _temp_artifact_dir(self):
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "weights.json").write_text(
            json.dumps({"w1": [[1.0]]}), encoding="utf-8"
        )
        return tmpdir

    def test_report_includes_required_fields(self):
        cand_dir = self._temp_artifact_dir()
        curr_dir = self._temp_artifact_dir()
        try:
            arena_results = [
                {
                    "budget_label": "standard",
                    "challenger_sims": 384,
                    "current_sims": 256,
                    "arena_score": 0.50,
                    "arena_wins": 60,
                    "arena_losses": 60,
                    "arena_draws": 0,
                    "seat_metrics": {
                        "disadvantaged_seat_score": 0.0,
                        "challenger_starts_0": {
                            "games": 60,
                            "wins": 60,
                            "losses": 0,
                            "draws": 0,
                            "score": 1.0,
                            "ci95": {"lower": 0.94, "upper": 1.0},
                            "margin_mean": 5.0,
                            "margin_median": 5.0,
                            "game_length_mean": 30.0,
                            "game_length_median": 30.0,
                        },
                        "challenger_starts_1": {
                            "games": 60,
                            "wins": 0,
                            "losses": 60,
                            "draws": 0,
                            "score": 0.0,
                            "ci95": {"lower": 0.0, "upper": 0.06},
                            "margin_mean": -5.0,
                            "margin_median": -5.0,
                            "game_length_mean": 30.0,
                            "game_length_median": 30.0,
                        },
                        "margin_mean": 0.0,
                        "margin_median": 0.0,
                        "game_length_mean": 30.0,
                        "game_length_median": 30.0,
                        "unique_trajectories": 2,
                        "duplicate_trajectory_count": 118,
                        "total_games": 120,
                        "first_move_challenger_dist": {"1": 8, "2": 7},
                        "first_move_current_dist": {"2": 8, "1": 7},
                    },
                    "move_time_mean_ms": 30.0,
                    "move_time_p95_ms": 50.0,
                },
                {
                    "budget_label": "equal_high",
                    "challenger_sims": 1200,
                    "current_sims": 1200,
                    "arena_score": 0.50,
                    "arena_wins": 60,
                    "arena_losses": 60,
                    "arena_draws": 0,
                    "seat_metrics": {
                        "disadvantaged_seat_score": 0.0,
                        "challenger_starts_0": {
                            "games": 60,
                            "wins": 60,
                            "losses": 0,
                            "draws": 0,
                            "score": 1.0,
                            "ci95": {"lower": 0.94, "upper": 1.0},
                            "margin_mean": 3.0,
                            "margin_median": 3.0,
                            "game_length_mean": 35.0,
                            "game_length_median": 35.0,
                        },
                        "challenger_starts_1": {
                            "games": 60,
                            "wins": 0,
                            "losses": 60,
                            "draws": 0,
                            "score": 0.0,
                            "ci95": {"lower": 0.0, "upper": 0.06},
                            "margin_mean": -3.0,
                            "margin_median": -3.0,
                            "game_length_mean": 35.0,
                            "game_length_median": 35.0,
                        },
                        "margin_mean": 0.0,
                        "margin_median": 0.0,
                        "game_length_mean": 35.0,
                        "game_length_median": 35.0,
                        "unique_trajectories": 2,
                        "duplicate_trajectory_count": 118,
                        "total_games": 120,
                        "first_move_challenger_dist": {"1": 8, "2": 7},
                        "first_move_current_dist": {"2": 8, "1": 7},
                    },
                    "move_time_mean_ms": 110.0,
                    "move_time_p95_ms": 170.0,
                },
            ]

            report = build_seat_aware_report(
                candidate_path=str(cand_dir),
                current_path=str(curr_dir),
                arena_results=arena_results,
            )

            self.assertEqual("azlite_seat_aware_promotion_gate_v1", report["schema"])
            self.assertIn("classification", report)
            self.assertIn("standard_alternating_score", report)
            self.assertIn("budget_results", report)
            self.assertIn("ranking_table", report)
            self.assertIn("candidate_sha256", report)
            self.assertIn("current_sha256", report)

            self.assertEqual("seat_artifact_only", report["classification"])
            self.assertEqual(0.50, report["standard_alternating_score"])

            rt = report["ranking_table"]
            self.assertIsNotNone(rt["standard_alternating_score"])
            self.assertEqual(0.0, rt["standard_disadvantaged_seat_score"])
            self.assertEqual("seat_artifact_only", rt["classification"])

        finally:
            import shutil

            shutil.rmtree(cand_dir)
            shutil.rmtree(curr_dir)

    def test_ranking_table_includes_all_columns(self):
        cand_dir = self._temp_artifact_dir()
        curr_dir = self._temp_artifact_dir()
        try:
            arena_results = [
                {
                    "budget_label": "standard",
                    "challenger_sims": 384,
                    "current_sims": 256,
                    "arena_score": 0.50,
                    "arena_wins": 60,
                    "arena_losses": 60,
                    "arena_draws": 0,
                    "seat_metrics": {
                        "disadvantaged_seat_score": 0.1,
                        "challenger_starts_0": {
                            "games": 60,
                            "wins": 60,
                            "losses": 0,
                            "draws": 0,
                            "score": 1.0,
                            "ci95": {"lower": 0.94, "upper": 1.0},
                            "margin_mean": 5.0,
                            "margin_median": 5.0,
                            "game_length_mean": 30.0,
                            "game_length_median": 30.0,
                        },
                        "challenger_starts_1": {
                            "games": 60,
                            "wins": 6,
                            "losses": 54,
                            "draws": 0,
                            "score": 0.1,
                            "ci95": {"lower": 0.04, "upper": 0.21},
                            "margin_mean": -3.0,
                            "margin_median": -3.0,
                            "game_length_mean": 30.0,
                            "game_length_median": 30.0,
                        },
                        "margin_mean": 1.0,
                        "margin_median": 1.0,
                        "game_length_mean": 30.0,
                        "game_length_median": 30.0,
                        "unique_trajectories": 30,
                        "duplicate_trajectory_count": 90,
                        "total_games": 120,
                        "first_move_challenger_dist": {"1": 8, "2": 7},
                        "first_move_current_dist": {"2": 8, "1": 7},
                    },
                    "move_time_mean_ms": 30.0,
                    "move_time_p95_ms": 50.0,
                },
            ]

            report = build_seat_aware_report(
                candidate_path=str(cand_dir),
                current_path=str(curr_dir),
                arena_results=arena_results,
            )

            rt = report["ranking_table"]
            required_cols = [
                "candidate",
                "artifact_sha256",
                "current_sha256",
                "standard_alternating_score",
                "standard_disadvantaged_seat_score",
                "equal_high_disadvantaged_seat_score",
                "challenger_high_disadvantaged_seat_score",
                "current_high_disadvantaged_seat_score",
                "margin_mean",
                "latency_p95_ms",
                "classification",
            ]
            for col in required_cols:
                self.assertIn(col, rt, f"missing column: {col}")

            self.assertEqual(0.1, rt["standard_disadvantaged_seat_score"])
        finally:
            import shutil

            shutil.rmtree(cand_dir)
            shutil.rmtree(curr_dir)


class BuildCandidateRankingTest(unittest.TestCase):
    def test_ranks_by_disadvantaged_seat_score(self):
        reports = [
            {
                "ranking_table": {
                    "candidate": "weak",
                    "standard_disadvantaged_seat_score": 0.0,
                    "equal_high_disadvantaged_seat_score": 0.0,
                    "challenger_high_disadvantaged_seat_score": 0.0,
                    "classification": "seat_artifact_only",
                }
            },
            {
                "ranking_table": {
                    "candidate": "strong",
                    "standard_disadvantaged_seat_score": 0.5,
                    "equal_high_disadvantaged_seat_score": 1.0,
                    "challenger_high_disadvantaged_seat_score": 1.0,
                    "classification": "standard_budget_breakthrough",
                }
            },
            {
                "ranking_table": {
                    "candidate": "medium",
                    "standard_disadvantaged_seat_score": 0.0,
                    "equal_high_disadvantaged_seat_score": 0.5,
                    "challenger_high_disadvantaged_seat_score": 0.0,
                    "classification": "high_search_breakthrough",
                }
            },
        ]

        ranking = build_candidate_ranking(reports)
        self.assertEqual(3, len(ranking))
        self.assertEqual("strong", ranking[0]["candidate"])
        self.assertEqual("medium", ranking[1]["candidate"])
        self.assertEqual("weak", ranking[2]["candidate"])

    def test_filters_out_empty_tables(self):
        reports = [
            {"ranking_table": {}},
            {"ranking_table": None},
        ]
        ranking = build_candidate_ranking(reports)
        self.assertEqual(0, len(ranking))


class Sha256FileTest(unittest.TestCase):
    def test_consistent_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()
            path = Path(f.name)

        try:
            h1 = sha256_file(path)
            h2 = sha256_file(path)
            self.assertEqual(h1, h2)
            self.assertEqual(64, len(h1))
        finally:
            os.unlink(path)


class BudgetPairLabelsTest(unittest.TestCase):
    def test_standard_budget_mapped(self):
        self.assertEqual("standard", BUDGET_PAIR_LABELS[(384, 256)])

    def test_equal_high_mapped(self):
        self.assertEqual("equal_high", BUDGET_PAIR_LABELS[(1200, 1200)])

    def test_challenger_high_mapped(self):
        self.assertEqual("challenger_high", BUDGET_PAIR_LABELS[(1200, 256)])

    def test_current_high_asymmetry_mapped(self):
        self.assertEqual("current_high_asymmetry", BUDGET_PAIR_LABELS[(256, 768)])


class AlternatingScoreBackwardCompatibilityTest(unittest.TestCase):
    def test_alternating_score_preserved_when_seat_artifact(self):
        entries = []
        for i in range(60):
            entries.append(
                _make_game_entry(
                    game_index=i * 2,
                    challenger_player=0,
                    winner="challenger",
                    margin=10,
                )
            )
        for i in range(60):
            entries.append(
                _make_game_entry(
                    game_index=i * 2 + 1,
                    challenger_player=1,
                    winner="current",
                    margin=-10,
                )
            )

        metrics = compute_seat_split_metrics(entries)
        p0_score = metrics["challenger_starts_0"]["score"]
        p1_score = metrics["challenger_starts_1"]["score"]
        total_wins = (
            metrics["challenger_starts_0"]["wins"]
            + metrics["challenger_starts_1"]["wins"]
        )
        total_draws = (
            metrics["challenger_starts_0"]["draws"]
            + metrics["challenger_starts_1"]["draws"]
        )
        total_games = metrics["total_games"]
        alternating_score = (total_wins + 0.5 * total_draws) / total_games

        self.assertAlmostEqual(1.0, p0_score)
        self.assertAlmostEqual(0.0, p1_score)
        self.assertAlmostEqual(0.50, alternating_score)

    def test_disadvantaged_seat_score_matches_p1_rail(self):
        entries = []
        for i in range(30):
            entries.append(
                _make_game_entry(
                    game_index=i, challenger_player=1, winner="challenger", margin=5
                )
            )
        for i in range(30):
            entries.append(
                _make_game_entry(
                    game_index=i + 30, challenger_player=1, winner="current", margin=-5
                )
            )

        metrics = compute_seat_split_metrics(entries)
        self.assertAlmostEqual(0.50, metrics["disadvantaged_seat_score"])
        self.assertAlmostEqual(0.50, metrics["challenger_starts_1"]["score"])


if __name__ == "__main__":
    unittest.main()
