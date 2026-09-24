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
    merge_gate_report,
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

    def test_repository_relative_artifact_takes_precedence_over_record_directory(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            record = new_record(generation_id="repository-relative")
            record["artifacts"].append(
                artifact_ref("config", root / "pyproject.toml", required=True)
            )
            record["artifacts"][0]["path"] = "pyproject.toml"
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

    def test_gate_merge_keeps_prefilter_pass_distinct_from_downstream_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(
                '{"arena_report_path":"arena.json","arena_score":0.6,'
                '"min_arena_score":0.55,"hard_report_path":"hard.json",'
                '"hard_score":0.5,"hard_min_score":0.55,"passed":false}',
                encoding="utf-8",
            )
            record = new_record(generation_id="gate")
            merge_gate_report(record, path)
            self.assertTrue(
                record["canonical_evaluation"]["prefilter"]["summary"]["passed"]
            )
            self.assertFalse(
                record["canonical_evaluation"]["hard_arena"]["summary"]["passed"]
            )

    def test_promotion_ready_does_not_mark_the_generation_promoted(self):
        record = new_record(generation_id="ready")
        record_promotion(record, decision="promotion_ready")
        self.assertEqual("promotion_ready", record["promotion"]["decision"])
        self.assertEqual("promotion_ready", record["status"])

    def test_historical_records_and_registry(self):
        root = Path(__file__).resolve().parents[2]
        for generation_id in (
            "seed48-incumbent",
            "seed48-nextgen-s443",
            "seed48-nextgen-s449",
            "seed48-nextgen-s455-default-value",
            "seed461-exact-root-optimal-set-uniform",
            "value-target-s401-sharpened",
            "value-target-s401-default",
            "value-target-s407-sharpened",
            "value-target-s407-default",
            "value-target-s413-sharpened",
            "value-target-s413-default",
            "value-target-s419-sharpened",
            "value-target-s419-default",
            "value-target-s443-sharpened",
            "value-target-s443-default",
            "value-target-s449-sharpened",
            "value-target-s449-default",
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
