from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import holdback_arena as protocol


def row(n: int) -> dict:
    return {"state": {"n": n}, "production_input": [float(n), 1.0]}


def setup_bundle(path, *, revealed=True):
    path.mkdir()
    pool = [row(i) for i in range(10)]
    excluded = {"rows": [row(99)]}
    generator = {"name": "synthetic-rules-only-v1", "seed": 17, "size": 4}
    suite = protocol.build_suite(pool, excluded["rows"], 17, 4)
    raw = protocol.suite_bytes(suite)
    (path / "suite.jsonl").write_bytes(raw)
    inv = protocol.canonical_bytes(excluded)
    (path / "inventory.json").write_bytes(inv)
    freeze = {
        "candidate_artifact_sha256": "a" * 64,
        "serving_p_artifact_sha256": "b" * 64,
        "arena_settings": {"pairs": 4, "workers": 1},
        "analysis_rule": "paired-opening mean",
    }
    (path / "candidate-freeze.json").write_text(json.dumps(freeze))
    manifest = {
        "suite_sha256": protocol.sha256(raw),
        "inventory_path": "inventory.json",
        "inventory_sha256": protocol.sha256(inv),
        "generator": generator,
        "commitment": protocol.commitment(raw, generator, excluded),
        "revealed": revealed,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    if revealed:
        (path / "consumed.json").write_text(
            json.dumps({"commitment": manifest["commitment"]})
        )
    return suite


class HoldbackProtocolTest(unittest.TestCase):
    def test_deterministic_rebuild_and_physical_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            source = tmp_path / "source"
            suite = setup_bundle(source)
            self.assertEqual(
                protocol.build_suite([row(i) for i in range(10)], [row(99)], 17, 4),
                suite,
            )
            copy = tmp_path / "copy"
            shutil.copytree(source, copy)
            shutil.rmtree(source)
            self.assertEqual(protocol.verify(copy)["openings"], 4)

    def test_duplicate_and_overlap_rejected_in_both_identity_schemes(self):
        with self.assertRaisesRegex(ValueError, "duplicate_exclusion"):
            protocol.validate_inventory([row(1), row(1)])
        # Different canonical states but identical float32 production representation.
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite(
                [row(1), {"state": {"n": 2}, "production_input": [1.0, 1.0]}], [], 1, 2
            )
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite([row(1)], [row(1)], 1, 1)

    def test_suite_bytes_order_and_consumed_reuse_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            path = tmp_path / "bundle"
            setup_bundle(path)
            lines = (path / "suite.jsonl").read_bytes().splitlines(keepends=True)
            (path / "suite.jsonl").write_bytes(b"".join(reversed(lines)))
            with self.assertRaisesRegex(ValueError, "suite_bytes_changed"):
                protocol.verify(path)
            consumed = tmp_path / "consumed"
            setup_bundle(consumed)
            self.assertEqual(protocol.verify(consumed)["status"], "verified")
            manifest = json.loads((consumed / "manifest.json").read_text())
            manifest["revealed"] = False
            (consumed / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "consumed_suite_not_revealed"):
                protocol.verify(consumed)

    def test_reveal_requires_candidate_and_serving_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle"
            setup_bundle(path)
            freeze = {"candidate_artifact_sha256": "a" * 64}
            (path / "candidate-freeze.json").write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError, "reveal_before_candidate_freeze"):
                protocol.verify(path)


if __name__ == "__main__":
    unittest.main()
