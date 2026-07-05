import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import run_residual_v3_opening_iteration0_preflight as preflight


class ResidualV3OpeningIteration0PreflightTest(unittest.TestCase):
    def test_select_candidates_sequential_path(self):
        entries = [
            {
                "state": {"current_player": 0},
                "prefix_moves": [0],
                "source_suite_path": "suite.jsonl",
                "source_suite_sha256": "suite-sha",
                "source_row_index": 0,
            }
        ]

        with (
            mock.patch.object(
                preflight.arena, "ArtifactEvaluator", return_value=object()
            ),
            mock.patch.object(
                preflight,
                "_evaluate_entry_with_evaluator",
                return_value={"state_hash": "picked"},
            ) as evaluate_mock,
        ):
            selected = preflight.select_candidates(
                entries=entries,
                current_path=Path("model-artifact/current"),
                search_profile=preflight.build_promoted_search_profile(),
                seed=42,
                workers=1,
            )

        self.assertEqual([{"state_hash": "picked"}], selected)
        evaluate_mock.assert_called_once()

    def test_select_candidates_parallel_path(self):
        entries = [
            {
                "state": {"current_player": 0},
                "prefix_moves": [0],
                "source_suite_path": "suite-a.jsonl",
                "source_suite_sha256": "suite-sha-a",
                "source_row_index": 0,
            },
            {
                "state": {"current_player": 1},
                "prefix_moves": [1],
                "source_suite_path": "suite-b.jsonl",
                "source_suite_sha256": "suite-sha-b",
                "source_row_index": 1,
            },
        ]

        class FakeExecutor:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, records):
                return map(func, records)

        with (
            mock.patch.object(preflight, "ProcessPoolExecutor", FakeExecutor),
            mock.patch.object(
                preflight,
                "_evaluate_entry_worker",
                side_effect=[{"state_hash": "a"}, None],
            ) as worker_mock,
        ):
            selected = preflight.select_candidates(
                entries=entries,
                current_path=Path("model-artifact/current"),
                search_profile=preflight.build_promoted_search_profile(),
                seed=42,
                workers=24,
            )

        self.assertEqual([{"state_hash": "a"}], selected)
        self.assertEqual(2, worker_mock.call_count)

    def test_selection_metrics_detects_instability_and_weakness(self):
        rows = [
            {
                "label": "sim256_default",
                "selected_move": 0,
                "top_share": 0.51,
                "margin": 0.06,
                "entropy": 1.42,
            },
            {
                "label": "sim384_default",
                "selected_move": 0,
                "top_share": 0.54,
                "margin": 0.09,
                "entropy": 1.39,
            },
            {
                "label": "sim768_default",
                "selected_move": 0,
                "top_share": 0.63,
                "margin": 0.17,
                "entropy": 1.11,
            },
            {
                "label": "sim768_equal_override",
                "selected_move": 2,
                "top_share": 0.50,
                "margin": 0.07,
                "entropy": 1.50,
            },
            {
                "label": "sim1200_default",
                "selected_move": 2,
                "top_share": 0.59,
                "margin": 0.14,
                "entropy": 1.21,
            },
        ]

        metrics = preflight.selection_metrics(rows)

        self.assertTrue(metrics["unstable"])
        self.assertTrue(metrics["weak"])
        self.assertTrue(metrics["equal_budget_override_move_flip"])
        self.assertEqual(2, metrics["move_set_size"])
        self.assertGreater(metrics["selection_score"], 0)

    def test_validate_guardrails_rejects_non_residual_v3_and_transforms(self):
        profile = preflight.build_promoted_search_profile()
        preflight.validate_guardrails(search_profile=profile, model_type="residual_v3")

        with self.assertRaisesRegex(RuntimeError, "model_type"):
            preflight.validate_guardrails(
                search_profile=profile,
                model_type="residual_v4_move_factorized",
            )

        bad_profile = dict(profile)
        bad_profile["value_transform"] = {"name": "danger"}
        with self.assertRaisesRegex(RuntimeError, "value_transform"):
            preflight.validate_guardrails(
                search_profile=bad_profile,
                model_type="residual_v3",
            )

    def test_deduplicate_by_state_is_deterministic(self):
        shared_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        entries = [
            {"state": shared_state, "prefix_moves": [0], "ply": 1},
            {"state": shared_state, "prefix_moves": [1], "ply": 1},
        ]

        deduped, stats = preflight.deduplicate_by_state(entries)

        self.assertEqual(1, len(deduped))
        self.assertEqual(1, stats["duplicate_state_rows_removed"])
        self.assertEqual(2, stats["unique_prefix_rows"])

    def test_manifest_is_reproducible_for_same_inputs(self):
        selected_rows = [
            {
                "state_hash": "a",
                "prefix_hash": "b",
                "phase_bucket": "early",
                "side_to_move_label": "0",
                "ply_label": "2",
                "first_move_family": "0",
                "primary_tag": "unstable_search",
            }
        ]
        search_profile = preflight.build_promoted_search_profile()

        with tempfile.TemporaryDirectory(prefix="azlite-preflight-") as tmp:
            tmp_path = Path(tmp)
            rows_path = tmp_path / "rows.jsonl"
            preflight.write_jsonl(rows_path, selected_rows)
            source_summaries = [
                {"path": "suite.jsonl", "sha256": "suite-sha", "rows": 1}
            ]
            dedup_stats = {
                "input_rows": 1,
                "unique_state_rows": 1,
                "duplicate_state_rows_removed": 0,
                "duplicate_prefix_rows_observed": 0,
                "unique_prefix_rows": 1,
                "state_fingerprint_sha256": "state-fp",
                "prefix_fingerprint_sha256": "prefix-fp",
            }

            first = preflight.build_manifest(
                current_path=Path("model-artifact/current"),
                current_weights_sha256="weights-sha",
                source_summaries=source_summaries,
                search_profile=search_profile,
                selected_rows_path=rows_path,
                selected_rows=selected_rows,
                selection_limit=1,
                dedup_stats=dedup_stats,
            )
            second = preflight.build_manifest(
                current_path=Path("model-artifact/current"),
                current_weights_sha256="weights-sha",
                source_summaries=source_summaries,
                search_profile=search_profile,
                selected_rows_path=rows_path,
                selected_rows=selected_rows,
                selection_limit=1,
                dedup_stats=dedup_stats,
            )

        self.assertEqual(first, second)
        self.assertEqual(
            first["search_profile_hash"],
            preflight.search_profile_hash(search_profile),
        )

    def test_main_writes_manifest_and_rows_deterministically(self):
        fake_search_rows = [
            {
                "label": "sim256_default",
                "budget_pair": "256:768",
                "simulations": 256,
                "role_context": ["384:256/current"],
                "c_puct": 1.25,
                "selected_move": 0,
                "legal_moves": [0, 1],
                "top_share": 0.51,
                "margin": 0.05,
                "entropy": 1.41,
                "root_value": 0.1,
                "search_policy": [0.51, 0.49, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "label": "sim384_default",
                "budget_pair": "384:256",
                "simulations": 384,
                "role_context": ["384:256/challenger"],
                "c_puct": 1.25,
                "selected_move": 0,
                "legal_moves": [0, 1],
                "top_share": 0.54,
                "margin": 0.09,
                "entropy": 1.31,
                "root_value": 0.1,
                "search_policy": [0.54, 0.46, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "label": "sim768_default",
                "budget_pair": "768:256",
                "simulations": 768,
                "role_context": ["768:256/challenger"],
                "c_puct": 1.25,
                "selected_move": 0,
                "legal_moves": [0, 1],
                "top_share": 0.67,
                "margin": 0.25,
                "entropy": 0.96,
                "root_value": 0.2,
                "search_policy": [0.67, 0.33, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "label": "sim768_equal_override",
                "budget_pair": "768:768",
                "simulations": 768,
                "role_context": ["768:768/shared"],
                "c_puct": 0.9,
                "selected_move": 1,
                "legal_moves": [0, 1],
                "top_share": 0.50,
                "margin": 0.00,
                "entropy": 1.5,
                "root_value": 0.2,
                "search_policy": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "label": "sim1200_default",
                "budget_pair": "1200:1200",
                "simulations": 1200,
                "role_context": ["1200:1200/shared"],
                "c_puct": 1.25,
                "selected_move": 1,
                "legal_moves": [0, 1],
                "top_share": 0.58,
                "margin": 0.16,
                "entropy": 1.2,
                "root_value": 0.3,
                "search_policy": [0.42, 0.58, 0.0, 0.0, 0.0, 0.0],
            },
        ]
        input_entries = [
            {
                "state": {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                "prefix_moves": [0, 1],
                "ply": 2,
                "phase_bucket": "early",
                "side_to_move": 0,
                "first_move_family": "0",
                "source_suite_path": "suite.jsonl",
                "source_suite_sha256": "suite-sha",
                "source_row_index": 0,
            }
        ]
        metadata = {
            "architecture": {"model_type": "residual_v3"},
        }

        with tempfile.TemporaryDirectory(prefix="azlite-preflight-main-") as tmp:
            tmp_path = Path(tmp)
            current_dir = tmp_path / "current"
            current_dir.mkdir()
            (current_dir / "weights.json").write_text("weights", encoding="utf-8")
            (current_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            expected_sha = preflight.sha256_file(current_dir / "weights.json")
            argv = [
                "prog",
                "--workdir",
                str(tmp_path / "work"),
                "--current",
                str(current_dir),
                "--expected-current-sha256",
                expected_sha,
                "--selection-limit",
                "1",
                "--max-input-rows",
                "1",
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch.object(
                    preflight,
                    "load_input_entries",
                    return_value=(
                        input_entries,
                        [{"path": "suite.jsonl", "sha256": "suite-sha", "rows": 1}],
                    ),
                ),
                mock.patch.object(
                    preflight.arena, "ArtifactEvaluator", return_value=object()
                ),
                mock.patch.object(
                    preflight,
                    "evaluate_search_setting",
                    side_effect=fake_search_rows,
                ),
            ):
                rc = preflight.main()

            self.assertEqual(0, rc)
            manifest_path = (
                tmp_path / "work/manifests/iteration0_training_manifest.json"
            )
            rows_path = tmp_path / "work/manifests/iteration0_selected_positions.jsonl"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(rows_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(preflight.PREFLIGHT_SCHEMA, manifest["schema"])
            self.assertEqual(
                preflight.PREFLIGHT_CLASSIFICATION,
                manifest["classification"],
            )
            rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual(1, rows[0]["selection_rank"])


if __name__ == "__main__":
    unittest.main()
