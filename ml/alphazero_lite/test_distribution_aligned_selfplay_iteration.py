"""Contract tests for the distribution-aligned replay ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.alphazero_lite.build_opening_suite import canonical_key
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import (
    build_training_opening_corpus,
    distribution_audit,
    opening_depth_feasibility,
    preflight,
    medium_strength_pass,
    suite_identities,
    truncate_matched_whole_games,
)


class DistributionAlignedSelfPlayIterationTest(unittest.TestCase):
    def test_corpus_is_unique_disjoint_and_includes_feasible_opening_depths(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite.jsonl"
            suite.write_text("", encoding="utf-8")
            corpus, manifest = build_training_opening_corpus(
                out_dir=root, exclusion_suites=[suite], seed=42, target_size=128
            )
            records = [json.loads(line) for line in corpus.read_text().splitlines()]
            self.assertEqual(128, manifest["state_count"])
            self.assertEqual(
                128, len({canonical_key(record["state"]) for record in records})
            )
            self.assertEqual({2, 4, 6}, set(manifest["prefix_depth_distribution"]))
            self.assertEqual({0, 1}, set(manifest["player_to_move_distribution"]))
            self.assertEqual(
                {"states": 0, "prefixes": 0, "alternate_prefixes": 0},
                manifest["exact_overlap_counts"],
            )

    def test_suite_identities_includes_alternate_prefixes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            state = {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
            path.write_text(
                json.dumps(
                    {
                        "state": state,
                        "prefix_moves": [1, 2],
                        "alternate_prefixes": [[3, 4]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            states, prefixes, hashes = suite_identities([path])
            self.assertEqual({canonical_key(state)}, states)
            self.assertEqual({(1, 2), (3, 4)}, prefixes)
            self.assertIn(str(path), hashes)

    def test_preflight_requires_distance_improvement_no_leakage_and_strata(
        self,
    ) -> None:
        control = {
            "seeded_start_state_overlap": 0,
            "seeded_prefix_overlap": 0,
            "nearest_board_l1": {"median": 10.0, "p90": 20.0},
            "player_distribution": {"0": 100, "1": 100},
            "phase_distribution": {"opening": 100},
        }
        aligned = {
            "seeded_start_state_overlap": 0,
            "seeded_prefix_overlap": 0,
            "nearest_board_l1": {"median": 7.0, "p90": 17.0},
            "player_distribution": {"0": 75, "1": 75},
            "phase_distribution": {"opening": 75},
        }
        self.assertTrue(preflight(control, aligned)[0])
        leaked = {**aligned, "seeded_start_state_overlap": 1}
        self.assertIn("seeded_start_state_leakage", preflight(control, leaked)[1])

    def test_distribution_audit_reports_trajectory_overlap_without_leakage(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            state = {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
            path.write_text(json.dumps({"state": state}) + "\n", encoding="utf-8")
            rows = [{"state": [4 / 48] * 12 + [0.0, 0.0, 0.0], "player": 0}]
            audit = distribution_audit(rows, [path])
            self.assertEqual(0, audit["seeded_start_state_overlap"])
            self.assertEqual(
                1,
                audit["replay_trajectory_state_overlap"][
                    "unique_overlapping_replay_states"
                ],
            )

    def test_feasibility_marks_exhausted_depths_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            # Enumerating the suite itself makes every ply-2 state unavailable.
            from ml.alphazero_lite.build_opening_suite import enumerate_legal_prefixes

            ply2 = [entry for entry in enumerate_legal_prefixes(2) if entry["ply"] == 2]
            path.write_text(
                "".join(json.dumps(entry) + "\n" for entry in ply2), encoding="utf-8"
            )
            report, feasible = opening_depth_feasibility([path])
            self.assertEqual(0, report["depths"]["2"]["eligible_after_all_exclusions"])
            self.assertNotIn(2, feasible)

    def test_matching_truncation_never_splits_a_game(self) -> None:
        def rows(lengths: tuple[int, ...]) -> list[dict]:
            return [
                {
                    "game_index": game,
                    "value": float(game % 2),
                    "seat_context": "player_0",
                    "game_length": length,
                }
                for game, length in enumerate(lengths)
                for _ in range(length)
            ]

        control = rows((3, 2, 4))
        aligned = rows((3, 2, 4))
        matched_control, matched_aligned = truncate_matched_whole_games(
            control, aligned, seed=42
        )
        self.assertEqual(len(matched_control), len(matched_aligned))
        self.assertEqual(
            [0, 0, 0, 1, 1, 2, 2, 2, 2], [row["game_index"] for row in matched_control]
        )
        self.assertEqual(
            [0, 0, 0, 1, 1, 2, 2, 2, 2], [row["game_index"] for row in matched_aligned]
        )

    def test_medium_gate_requires_paired_lower_bound_and_robustness(self) -> None:
        def ci(mean: float, lower: float = 0.01) -> dict:
            return {"mean": mean, "lower": lower}

        screen = {
            "paired_opening_bootstrap_95": {
                "aligned_minus_control": {
                    "384:256": ci(0.03),
                    "768:768": ci(-0.02),
                    "1200:1200": ci(-0.02),
                    "1200:256": ci(-0.02),
                },
                "aligned_minus_current": {
                    "384:256": ci(0.01),
                    "768:768": ci(-0.03),
                    "1200:1200": ci(-0.03),
                    "1200:256": ci(-0.03),
                },
            }
        }
        self.assertTrue(medium_strength_pass(screen))
        screen["paired_opening_bootstrap_95"]["aligned_minus_control"]["384:256"][
            "lower"
        ] = 0.0
        self.assertFalse(medium_strength_pass(screen))


if __name__ == "__main__":
    unittest.main()
