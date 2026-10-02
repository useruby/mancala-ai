"""Regression checks for the seed397 evidence erratum and preserved #396."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.audit_seed397_evidence_erratum import build_audit
from ml.alphazero_lite.opening_exclusion_contract import (
    create_manifest,
    validate_suite_against_manifest,
)
from ml.alphazero_lite.build_opening_suite import load_suite_jsonl
from ml.alphazero_lite.reproduce_seed461_e1_e4_diagnostic import reproduce

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed461-e1-e4-corrected-diagnostic"


class Seed397ErratumTests(unittest.TestCase):
    def test_audit_reports_omissions_collisions_and_cross_evaluation_overlap(
        self,
    ) -> None:
        report = build_audit()
        self.assertEqual(
            report["excluded_state_counts"]["complete_registered_union"], 93366
        )
        self.assertEqual(report["excluded_state_counts"]["seed397_union"], 90832)
        self.assertEqual(report["excluded_state_counts"]["omitted_count"], 2534)
        self.assertEqual(
            report["historical_declared_collisions"]["opening_indices_zero_based"],
            [18, 107, 136, 137, 170, 262, 297, 336, 350, 385, 404, 503, 507],
        )
        self.assertEqual(report["historical_declared_collisions"]["count"], 13)
        self.assertEqual(report["shared_between_valid_396_and_invalid_397"]["count"], 7)

    def test_valid_396_suite_passes_complete_registered_manifest(self) -> None:
        manifest = json.loads(
            (DATA / "opening-exclusion-manifest.json").read_text(encoding="utf-8")
        )
        rows = load_suite_jsonl(str(DATA / "seed397-openings-v2.jsonl"))
        identities = validate_suite_against_manifest(rows, manifest)
        self.assertEqual(len(identities), 512)

    def test_actual_only_manifest_is_rejected_even_when_counts_match(self) -> None:
        full_manifest = json.loads(
            (DATA / "opening-exclusion-manifest.json").read_text(encoding="utf-8")
        )
        source = next(
            row for row in full_manifest["sources"] if row["kind"] == "historical_suite"
        )
        manifest = create_manifest([{"path": source["path"], "kind": source["kind"]}])
        manifest["declared_state_identities"] = manifest["actual_state_identities"]
        manifest["declared_state_count"] = manifest["actual_state_count"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            from ml.alphazero_lite.opening_exclusion_contract import verify_manifest

            with self.assertRaisesRegex(
                ValueError, "exclusion_manifest_stale_or_mutated"
            ):
                verify_manifest(path)

    def test_retired_cli_rejects_before_subprocess_execution(self) -> None:
        script = Path(__file__).with_name("run_seed397_o0_e1_e4.py")
        with patch.object(sys, "argv", [str(script), "run"]):
            with patch.object(subprocess, "run") as launch:
                with self.assertRaisesRegex(SystemExit, "seed397_protocol_invalid"):
                    runpy.run_path(str(script), run_name="__main__")
                launch.assert_not_called()

    def test_valid_396_scores_and_intervals_reproduce(self) -> None:
        matrix = DATA / "paired-opening-score-matrix.json"
        result = reproduce(matrix)
        self.assertEqual(result["means"]["E1"], 0.48388671875)
        self.assertEqual(result["means"]["E4"], 0.51171875)
        self.assertEqual(
            result["intervals_95"]["E1"], [0.46336669921875, 0.50439453125]
        )
        self.assertEqual(result["intervals_95"]["E4"], [0.49072265625, 0.53271484375])


if __name__ == "__main__":
    unittest.main()
