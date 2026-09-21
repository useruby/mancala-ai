import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.generation_record import (
    GenerationRecordError,
    artifact_ref,
    attach_diagnostic,
    load_record,
    migrate_manifest,
    new_record,
    record_candidate,
    record_promotion,
    validate,
    validate_index,
    write_record,
)


class GenerationRecordTest(unittest.TestCase):
    def test_planned_round_trip_and_unknown_schema(self):
        record = new_record(generation_id="planned")
        validate(record)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            write_record(path, record)
            self.assertEqual(record, load_record(path))
        record["schema"] = "future"
        with self.assertRaisesRegex(GenerationRecordError, "unsupported"):
            validate(record)

    def test_artifact_hash_and_required_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"
            path.write_text("original", encoding="utf-8")
            record = new_record(generation_id="artifact")
            record["artifacts"].append(artifact_ref("data", path, required=True))
            validate(record)
            path.write_text("mutated", encoding="utf-8")
            with self.assertRaisesRegex(GenerationRecordError, "mismatch"):
                validate(record)
            record["artifacts"] = [
                artifact_ref("optional", "unavailable", required=False)
            ]
            validate(record, base_dir=Path(temporary))
            record["artifacts"] = [
                artifact_ref("required", "unavailable", required=True)
            ]
            with self.assertRaisesRegex(GenerationRecordError, "lacks SHA256"):
                validate(record, base_dir=Path(temporary))

    def test_lifecycle_and_gate_identity(self):
        record = new_record(
            generation_id="candidate",
            parent={
                "version": "p",
                "weights_sha256": "a" * 64,
                "metadata_sha256": None,
            },
        )
        record_candidate(
            record,
            version="c",
            checkpoint={"role": "checkpoint", "path": "c", "sha256": "b" * 64},
            weights={"role": "weights", "path": "w", "sha256": "c" * 64},
            metadata={"role": "metadata", "path": "m", "sha256": "d" * 64},
        )
        validate(record)
        record_promotion(
            record, decision="rejected", failure_reasons=[{"code": "prefilter"}]
        )
        validate(record)
        record["promotion"]["gate_report"] = {
            "role": "gate_report",
            "path": "g",
            "sha256": "e" * 64,
            "candidate_weights_sha256": "wrong",
        }
        with self.assertRaisesRegex(GenerationRecordError, "does not match"):
            validate(record)

    def test_generic_diagnostics_and_manifest_migration(self):
        record = new_record(generation_id="diagnostics")
        attach_diagnostic(
            record, "future_metric", tool="future", config={}, summary={"score": 1}
        )
        validate(record)
        migrated = migrate_manifest(
            {
                "schema": "azlite_run_manifest_v1",
                "run_id": "x",
                "iteration": 2,
                "parent_version": "p",
            }
        )
        self.assertEqual("x-iter2", migrated["generation_id"])

    def test_historical_records_and_registry(self):
        root = Path(__file__).resolve().parents[2]
        for generation_id in (
            "seed48-incumbent",
            "seed48-nextgen-s443",
            "seed48-nextgen-s449",
        ):
            load_record(
                root
                / "docs/data/alphazero-lite-generations"
                / generation_id
                / "generation.json"
            )
        validate_index(root / "docs/data/alphazero-lite-generations/index.json")


if __name__ == "__main__":
    unittest.main()
