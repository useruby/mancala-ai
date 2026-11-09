"""Integrity tests for the seed558 paired value diagnostic package."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from ml.alphazero_lite.seed558_value_only_swap import ARMS, DATA, ROOT, run, score
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
        frozen = {
            name: (DATA / name).read_bytes()
            for name in (
                "execution-freeze.json",
                "search-records.jsonl",
                "analysis.json",
            )
        }
        self.assertEqual(verify(DATA)["status"], "verified")
        self.assertEqual({name: (DATA / name).read_bytes() for name in frozen}, frozen)
        copy = _copy()
        try:
            # The verifier consumes only the copied package and bundled artifacts.
            copied_records = (copy / "search-records.jsonl").read_bytes()
            self.assertEqual(verify(copy)["paired_searches"], 512)
            self.assertEqual(
                (copy / "search-records.jsonl").read_bytes(), copied_records
            )
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

    def test_swapped_arm_labels_with_regenerated_metrics_fail_replay(self) -> None:
        copy = _copy()
        try:
            searches = [
                json.loads(line)
                for line in (copy / "search-records.jsonl").read_text().splitlines()
            ]
            for row in searches:
                row["arm"] = next(arm for arm in ARMS if arm != row["arm"])
            (copy / "search-records.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
            )
            cohort = json.loads((copy / "cohort.json").read_text())["states"]
            oracles = [
                json.loads(line)
                for line in (copy / "oracle-records.jsonl").read_text().splitlines()
            ]
            rebuilt = score(cohort, searches, oracles)
            (copy / "root-metrics.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in rebuilt.pop("root_records")
                )
            )
            (copy / "analysis.json").write_text(
                json.dumps(rebuilt, sort_keys=True, indent=2) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "search_result_replay_mismatch"):
                verify(copy)
        finally:
            shutil.rmtree(copy)

    def test_existing_output_is_never_overwritten(self) -> None:
        frozen = (DATA / "execution-freeze.json").read_bytes()
        searches = (DATA / "search-records.jsonl").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "must be new"):
            run(DATA)
        self.assertEqual((DATA / "execution-freeze.json").read_bytes(), frozen)
        self.assertEqual((DATA / "search-records.jsonl").read_bytes(), searches)
