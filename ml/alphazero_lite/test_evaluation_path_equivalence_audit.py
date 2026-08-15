import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.evaluation_seed_contract import stable_hash
from ml.alphazero_lite.run_canonical_policy_interpolation_reconciliation import (
    cache_manifest,
    cache_status,
    run_search,
)


class CanonicalParallelBenchmarkCacheTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cache = Path(self.tempdir.name) / "records.json"
        self.manifest = cache_manifest(
            suite_sha256="suite-a",
            challenger_artifact_sha256="candidate-a",
            opponent_artifact_sha256="current-a",
            budget_pairs=("384:256",),
            c_puct_schedule={"default": 1.25, "768:768": 0.9},
            tactical_root_bias=0.0,
            root_policy="deterministic",
            seed_contract="azlite_eval_seed_v2",
            base_seed=42,
            games_per_opening=1,
        )
        self.records = [{"opening_index": 0, "score": 0.5}]
        self.cache.write_text(
            json.dumps(
                {
                    "manifest": self.manifest,
                    "records": self.records,
                    "records_sha256": stable_hash(self.records),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def assert_invalid(self, **changes):
        expected = {**self.manifest, **changes}
        self.assertEqual(
            (False, "manifest_mismatch"), cache_status(self.cache, expected)
        )

    def test_matching_manifest_and_output_reuses(self):
        self.assertEqual((True, "reused"), cache_status(self.cache, self.manifest))

    def test_stale_suite_invalidates(self):
        self.assert_invalid(suite_sha256="suite-b")

    def test_stale_artifact_invalidates(self):
        self.assert_invalid(challenger_artifact_sha256="candidate-b")

    def test_stale_cpuct_invalidates(self):
        self.assert_invalid(c_puct_resolution={"default": 1.1})

    def test_stale_seed_contract_invalidates(self):
        self.assert_invalid(seed_contract="azlite_eval_seed_v1")

    def test_corrupted_output_invalidates(self):
        payload = json.loads(self.cache.read_text(encoding="utf-8"))
        payload["records"][0]["score"] = 1.0
        self.cache.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(
            (False, "output_hash_mismatch"), cache_status(self.cache, self.manifest)
        )

    def test_canonical_search_delegates_to_arena_primitive(self):
        arena_result = {
            "selected_move": 2,
            "visits": [0.0, 1.0, 5.0, 0.0, 0.0, 0.0],
            "value": 0.25,
            "search_root_value": 0.5,
        }
        with patch(
            "ml.alphazero_lite.run_canonical_policy_interpolation_reconciliation.evaluate_artifact_position",
            return_value=arena_result,
        ) as evaluate:
            result = run_search(object(), {"current_player": 0}, 384, 123, 1.25)

        self.assertEqual(2, result["move"])
        self.assertEqual([0, 1, 5, 0, 0, 0], result["visits"])
        self.assertEqual(0.5, result["root_value"])
        self.assertEqual(1, evaluate.call_count)
        self.assertEqual(
            "deterministic",
            evaluate.call_args.kwargs["search_options"]["root_policy_mode"],
        )


if __name__ == "__main__":
    unittest.main()
