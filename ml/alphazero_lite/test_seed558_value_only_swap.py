"""Integrity tests for the seed558 paired value diagnostic package."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from ml.alphazero_lite.seed558_value_only_swap import DATA, ROOT
from ml.alphazero_lite.verify_seed558_value_only_swap import verify


def _copy() -> Path:
    destination = ROOT / ".tmp" / f"seed558-verifier-test-{uuid.uuid4().hex}"
    shutil.copytree(DATA, destination)
    return destination


def _edit_first_record(path: Path, update) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    update(rows[0])
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class Seed558ValueOnlySwapTests(unittest.TestCase):
    def test_fresh_package_and_physical_copy_verify(self) -> None:
        self.assertEqual(verify(DATA)["status"], "verified")
        copy = _copy()
        try:
            # The verifier consumes only the copied package and bundled artifacts.
            self.assertEqual(verify(copy)["paired_searches"], 512)
        finally:
            shutil.rmtree(copy)

    def test_changed_priors_fail(self) -> None:
        copy = _copy()
        try:
            _edit_first_record(
                copy / "search-records.jsonl",
                lambda row: row["priors"].__setitem__(0, row["priors"][0] + 0.01),
            )
            with self.assertRaisesRegex(ValueError, "priors"):
                verify(copy)
        finally:
            shutil.rmtree(copy)

    def test_selected_action_contradicted_by_visits_fails(self) -> None:
        copy = _copy()

        def contradict(row: dict) -> None:
            row["selected_action"] = next(
                move
                for move in row["legal_moves"]
                if move != row["selected_action"]
                and row["visits"][move]
                < max(row["visits"][m] for m in row["legal_moves"])
            )

        try:
            _edit_first_record(copy / "search-records.jsonl", contradict)
            with self.assertRaisesRegex(ValueError, "selected_action"):
                verify(copy)
        finally:
            shutil.rmtree(copy)

    def test_swapped_value_artifact_fails(self) -> None:
        copy = _copy()
        try:
            shutil.copyfile(
                copy / "artifacts/initializer/model.npz",
                copy / "artifacts/D432/model.npz",
            )
            with self.assertRaisesRegex(
                ValueError,
                "component_artifact_hash_mismatch|D432_artifact_identity_mismatch",
            ):
                verify(copy)
        finally:
            shutil.rmtree(copy)
