import argparse
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.run_canonical_runtime_profile_revalidation import (
    PROFILE_D,
    PROFILE_E,
    orientation_cache_identity,
    orientation_cache_status,
    write_orientation_cache_manifest,
)


class OrientationCacheManifestTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.suite = self.root / "suite.jsonl"
        self.suite.write_text('{"prefix_moves": []}\n', encoding="utf-8")
        self.seat_dir = self.root / "starts_0"
        self.seat_dir.mkdir()
        self.report = self.seat_dir / "arena.json"
        self.games = self.seat_dir / "games.jsonl"
        self.ledgers = {}
        for name in (
            "seed_identity_ledger",
            "search_configuration_ledger",
            "search_outcome_ledger",
        ):
            path = self.seat_dir / f"{name}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            self.ledgers[name] = path
        self.report.write_text(
            json.dumps(
                {
                    "notes": {
                        "seed_ledger_output": str(self.ledgers["seed_identity_ledger"]),
                        "search_configuration_ledger_output": str(
                            self.ledgers["search_configuration_ledger"]
                        ),
                        "search_outcome_ledger_output": str(
                            self.ledgers["search_outcome_ledger"]
                        ),
                    }
                }
            ),
            encoding="utf-8",
        )
        self.games.write_text('{"game_index": 0}\n', encoding="utf-8")
        self.manifest = self.seat_dir / "cache_manifest.json"
        self.args = argparse.Namespace(
            expected_current_weights_sha256="artifact-sha",
            seed_contract="azlite_eval_seed_v2",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def identity(self, *, budget="768:256"):
        return orientation_cache_identity(
            args=self.args,
            first=PROFILE_E,
            second=PROFILE_D,
            budget=budget,
            suite=self.suite,
            seed=42,
            seat=0,
            suite_size=1,
        )

    def test_manifest_with_matching_outputs_is_reused(self):
        identity = self.identity()
        write_orientation_cache_manifest(
            self.manifest, identity, self.report, self.games
        )

        reusable, reason, games_path = orientation_cache_status(
            self.manifest, identity, self.seat_dir
        )

        self.assertTrue(reusable)
        self.assertEqual("reused", reason)
        self.assertEqual(self.games, games_path)

    def test_manifest_identity_change_invalidates_cache(self):
        write_orientation_cache_manifest(
            self.manifest, self.identity(), self.report, self.games
        )

        reusable, reason, games_path = orientation_cache_status(
            self.manifest, self.identity(budget="1200:256"), self.seat_dir
        )

        self.assertFalse(reusable)
        self.assertEqual("manifest_identity_mismatch", reason)
        self.assertIsNone(games_path)

    def test_modified_cached_output_invalidates_cache(self):
        identity = self.identity()
        write_orientation_cache_manifest(
            self.manifest, identity, self.report, self.games
        )
        self.games.write_text('{"game_index": 1}\n', encoding="utf-8")

        reusable, reason, games_path = orientation_cache_status(
            self.manifest, identity, self.seat_dir
        )

        self.assertFalse(reusable)
        self.assertEqual("cached_games_hash_mismatch", reason)
        self.assertIsNone(games_path)

    def test_old_cache_without_manifest_is_diagnosed(self):
        reusable, reason, games_path = orientation_cache_status(
            self.manifest, self.identity(), self.seat_dir
        )

        self.assertFalse(reusable)
        self.assertEqual("legacy_cache_without_manifest", reason)
        self.assertIsNone(games_path)

    def test_missing_ledger_invalidates_cache(self):
        identity = self.identity()
        write_orientation_cache_manifest(
            self.manifest, identity, self.report, self.games
        )
        self.ledgers["seed_identity_ledger"].unlink()

        reusable, reason, games_path = orientation_cache_status(
            self.manifest, identity, self.seat_dir
        )

        self.assertFalse(reusable)
        self.assertEqual("cached_ledger_missing", reason)
        self.assertIsNone(games_path)


if __name__ == "__main__":
    unittest.main()
