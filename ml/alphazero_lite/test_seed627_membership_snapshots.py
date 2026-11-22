"""Synthetic checks for Seed 627 snapshot identity and immutability gates."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from ml.alphazero_lite import seed627_membership_snapshots as producer
from ml.alphazero_lite import verify_seed627_membership_snapshots as verifier


def row(index: int) -> dict:
    state = {
        "current_player": 0,
        "opponent_pits": [0, 0, 0, 0, 0, 0],
        "opponent_store": 0,
        "player_pits": [index, 0, 0, 0, 0, 0],
        "player_store": 0,
    }
    return {
        "canonical_state": state,
        "canonical_identity": producer.audit.identity(state),
        "production_float32_identity": producer.audit.float32_identity(state),
        "source_membership_index": index,
        "lineage_evidence": {"population": "test"},
    }


class SnapshotIdentityTests(unittest.TestCase):
    def populations(self) -> dict[str, list[dict]]:
        return {
            name: sorted(
                (row(index) for index in range(1, count + 1)),
                key=lambda item: producer.canonical(item["canonical_state"]),
            )
            for name, count in (
                ("B512", 2048),
                ("S615", 2048),
                ("protected_holdout", 1353),
            )
        }

    def test_canonical_and_float32_identity_recomputed(self) -> None:
        for item in (row(1), row(2)):
            self.assertEqual(
                producer.audit.identity(item["canonical_state"]),
                item["canonical_identity"],
            )
            self.assertEqual(
                producer.audit.float32_identity(item["canonical_state"]),
                item["production_float32_identity"],
            )

    def test_detects_canonical_mutation(self) -> None:
        item = row(1)
        item["canonical_state"]["player_store"] = 2
        populations = self.populations()
        populations["B512"][0] = item
        with self.assertRaisesRegex(ValueError, "snapshot_identity_mismatch"):
            producer.verify_snapshots(populations, {"B512": {}, "S615": {}})

    def test_create_only_publisher_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot"
            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "create_only"):
                producer.publish(destination)

    def test_rejects_missing_training_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_lineage_missing"):
            producer.verify_snapshots(self.populations(), {"B512": {}})

    def test_rejects_duplicate_snapshot_records(self) -> None:
        populations = self.populations()
        populations["B512"][1] = populations["B512"][0]
        with self.assertRaisesRegex(
            ValueError, "snapshot_sort_order_mismatch|snapshot_duplicate_identity"
        ):
            producer.verify_snapshots(
                populations, {"B512": {}, "S615": {}, "protected_holdout": {}}
            )

    def test_rejects_tampered_snapshot_bytes(self) -> None:
        expected = [row(1)]
        manifest = {
            "snapshot_sha256": {"B512": producer.digest(producer.canonical(expected))},
            "snapshot_file_sha256": {
                "B512": producer.digest(producer.canonical(expected) + b"\n")
            },
        }
        with self.assertRaisesRegex(
            ValueError, "snapshot_digest_mismatch|snapshot_rows"
        ):
            verifier.validate_snapshot_payload(
                "B512", producer.canonical([row(2)]) + b"\n", expected, manifest
            )

    def test_rejects_unproven_provenance_crosswalk(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_provenance_crosswalk"):
            verifier.validate_provenance(
                {
                    "source_game_opening": "authenticated_but_non_crosswalkable",
                    "crosswalk_rows": [{"opening_identity": "claimed"}],
                    "missing_or_insufficient": "not proven",
                    "inherited_PE4_ancestry": "unresolved_identity_incomplete_training_ledger",
                }
            )

    def test_missing_historical_verifier_is_not_approximated(self) -> None:
        with mock.patch.object(
            verifier.subprocess, "run", side_effect=FileNotFoundError
        ):
            with self.assertRaises(FileNotFoundError):
                verifier.rerun_historical_verifiers()

    def test_consumed_indexes_reconstructed_from_schedule_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / verifier.S599 / "training-run-B").mkdir(parents=True)
            schedule = [
                list(range((update * 32) % 2048, (update * 32) % 2048 + 32))
                for update in range(512)
            ]
            trace = b"".join(
                producer.canonical({"update": update, "indexes": batch}) + b"\n"
                for update, batch in enumerate(schedule, 1)
            )
            (root / verifier.S599 / "batch-schedule-B.json").write_bytes(
                producer.canonical(schedule)
            )
            (root / verifier.S599 / "training-execution-freeze.json").write_bytes(
                b"frozen"
            )
            output = root / verifier.S599 / "training-run-B"
            (output / "training-trace.jsonl").write_bytes(trace)
            receipt = {
                "trace_sha256": verifier.sha(trace),
                "updates": 512,
                "membership_sha256": verifier.EXPECTED["B512"],
                "parent_checkpoint_sha256": verifier.EXPECTED["parent"],
                "batch_schedule_sha256": verifier.sha(verifier.canonical(schedule)),
                "execution_freeze_sha256": verifier.sha(b"frozen"),
                "checkpoint_sha256": verifier.EXPECTED["B512_checkpoint"],
            }
            (output / "training-receipt.json").write_text(
                producer.canonical(receipt).decode()
            )
            overrides = {
                "B512_trace": verifier.sha(trace),
                "B512_freeze": verifier.sha(b"frozen"),
            }
            with (
                mock.patch.object(verifier, "ROOT", root),
                mock.patch.dict(verifier.EXPECTED, overrides),
            ):
                indexes, lineage = verifier.reconstruct_consumed("B512", [{}] * 2048)
            self.assertEqual(indexes, list(range(2048)))
            self.assertEqual(lineage["updates"], 512)
            self.assertEqual(lineage["unique_membership_rows_consumed"], 2048)

    def test_incorrect_training_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / verifier.S599 / "training-run-B").mkdir(parents=True)
            schedule = [
                list(range((update * 32) % 2048, (update * 32) % 2048 + 32))
                for update in range(512)
            ]
            trace_rows = [
                {"update": update, "indexes": list(batch)}
                for update, batch in enumerate(schedule, 1)
            ]
            trace_rows[0]["indexes"][0] = 2047
            trace = b"".join(producer.canonical(item) + b"\n" for item in trace_rows)
            (root / verifier.S599 / "batch-schedule-B.json").write_bytes(
                producer.canonical(schedule)
            )
            (root / verifier.S599 / "training-execution-freeze.json").write_bytes(
                b"frozen"
            )
            output = root / verifier.S599 / "training-run-B"
            (output / "training-trace.jsonl").write_bytes(trace)
            (output / "training-receipt.json").write_text("{}")
            with (
                mock.patch.object(verifier, "ROOT", root),
                mock.patch.dict(
                    verifier.EXPECTED,
                    {
                        "B512_trace": verifier.sha(trace),
                        "B512_freeze": verifier.sha(b"frozen"),
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "trace_schedule_mismatch"):
                    verifier.reconstruct_consumed("B512", [{}] * 2048)

    def test_missing_training_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / verifier.S599 / "training-run-B").mkdir(parents=True)
            with mock.patch.object(verifier, "ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    verifier.reconstruct_consumed("B512", [{}] * 2048)


if __name__ == "__main__":
    unittest.main()
