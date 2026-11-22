"""Prepare authenticated, read-only Seed 627 membership snapshots."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed622_training_provenance_audit as audit
from ml.alphazero_lite import verify_seed627_membership_snapshots as independent

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed627-authenticated-membership/snapshots"
SCHEMA = "seed627-authenticated-membership-snapshot-v1"
MANIFEST_SCHEMA = "seed627-source-identity-lineage-manifest-v1"
VERIFIER = "ml.alphazero_lite.verify_seed627_membership_snapshots"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _record(row: dict[str, Any], index: int, population: str) -> dict[str, Any]:
    state = row["canonical_state"]
    cid = audit.identity(state)
    fid = audit.float32_identity(state)
    if row.get("canonical_identity") != cid:
        raise ValueError(f"canonical_source_identity_mismatch:{population}:{index}")
    declared_float = row.get("float32_identity")
    if declared_float is not None and declared_float != fid:
        raise ValueError(f"float32_source_identity_mismatch:{population}:{index}")
    return {
        "canonical_state": state,
        "canonical_identity": cid,
        "production_float32_identity": fid,
        "source_membership_index": index,
        "lineage_evidence": {
            "population": population,
            "membership_source": (
                "docs/data/seed599-replay-coverage/treatment-membership.json"
                if population == "B512"
                else "docs/data/seed615-source-opening-diversity/candidate-membership.jsonl"
                if population == "S615"
                else "docs/data/seed599-replay-coverage/heldout-membership.json"
            ),
            "consumed_by_training": population in {"B512", "S615"},
            "training_updates": 512 if population in {"B512", "S615"} else None,
            "registered_batch_size": 32 if population in {"B512", "S615"} else None,
        },
    }


def reconstruct_snapshots() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Authenticate historical records and reconstruct exactly consumed rows."""
    populations, lineage = audit.authenticated_training()
    source_rows = {
        "B512": populations["B512"],
        "S615": populations["S615"],
        "protected_holdout": populations["protected_holdout"],
    }
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for name, rows in source_rows.items():
        records = [_record(row, index, name) for index, row in enumerate(rows)]
        records.sort(key=lambda row: canonical(row["canonical_state"]))
        cids = [row["canonical_identity"] for row in records]
        fids = [row["production_float32_identity"] for row in records]
        if name != "protected_holdout" and (
            len(cids) != len(set(cids)) or len(fids) != len(set(fids))
        ):
            raise ValueError(f"duplicate_training_identity:{name}")
        snapshots[name] = records
    return snapshots, lineage


def _raw_input_hashes() -> dict[str, str]:
    paths = {
        "B512_membership": audit.S599 / "treatment-membership.json",
        "B512_schedule": audit.S599 / "batch-schedule-B.json",
        "B512_trace": audit.S599 / "training-run-B/training-trace.jsonl",
        "B512_receipt": audit.S599 / "training-run-B/training-receipt.json",
        "B512_freeze": audit.S599 / "training-execution-freeze.json",
        "B512_independent_verification": audit.S599
        / "training-run-B/independent-verification.json",
        "S615_membership": audit.S615 / "candidate-membership.jsonl",
        "S615_trace": audit.S615 / "training-run-S615/training-trace.jsonl",
        "S615_receipt": audit.S615 / "training-run-S615/training-receipt.json",
        "S615_freeze": audit.S615 / "training-execution-freeze-v3.json",
        "S615_integrity_report": audit.S615
        / "training-run-S615/training-integrity-report.json",
        "S615_runtime_certification": audit.S615
        / "training-runtime-certification-v3.json",
        "S615_independent_verification": audit.S615 / "independent-verification.json",
        "protected_holdout": audit.S599 / "heldout-membership.json",
        "shared_parent_checkpoint": audit.PARENT,
        "seed599_preparation": audit.S599 / "preparation.json",
    }
    return {name: digest(path.read_bytes()) for name, path in paths.items()}


def _run_historical_verifiers() -> dict[str, dict[str, Any]]:
    commands = [
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed599_training_result",
            "--arm",
            "B",
            "--output-dir",
            "docs/data/seed599-replay-coverage/training-run-B",
        ],
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed615_training_result",
            "--output-dir",
            "docs/data/seed615-source-opening-diversity/training-run-S615",
        ],
        [
            sys.executable,
            "-c",
            "from ml.alphazero_lite.verify_seed615_training_freeze import verify; "
            "import json; print(json.dumps(verify(allow_execution_artifacts=True)))",
        ],
        [sys.executable, "-m", "ml.alphazero_lite.verify_seed615_opening_diversity"],
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed622_training_provenance_audit",
        ],
    ]
    results = {}
    for command in commands:
        completed = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
        results[command[2] if len(command) > 2 else command[1]] = json.loads(
            completed.stdout
        )
    return results


def publish(output: Path = OUT) -> dict[str, Any]:
    """Run pinned source verifiers and exclusive-create authenticated evidence."""
    if output.exists():
        raise FileExistsError("snapshot_destination_must_be_create_only")
    verifier_results = _run_historical_verifiers()
    snapshots, lineage = reconstruct_snapshots()
    # A separate verifier reconstructs source indexes and row identities before
    # any artifact is written; self-declared status fields are not accepted.
    independently_reconstructed, independent_lineage, _input_hashes = (
        independent.reconstruct()
    )
    if independently_reconstructed != snapshots or independent_lineage != lineage:
        raise ValueError("independent_source_reconstruction_disagrees")
    verification = verify_snapshots(snapshots, lineage)
    output.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, str] = {}
    file_outputs: dict[str, str] = {}
    for name, rows in snapshots.items():
        raw = canonical(rows) + b"\n"
        path = output / f"{name}.json"
        with path.open("xb") as stream:
            stream.write(raw)
        outputs[name] = digest(canonical(rows))
        file_outputs[name] = digest(raw)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "authenticated",
        "historical_evidence_modified": False,
        "runtime": {
            "python": sys.version,
            "numpy": audit.np.__version__,
            "torch": importlib.metadata.version("torch"),
        },
        "dependency_identity": {
            "runtime_lock": "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt",
            "runtime_lock_sha256": digest(
                (
                    ROOT
                    / "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
                ).read_bytes()
            ),
            "source_sha256": {
                path: digest((ROOT / path).read_bytes())
                for path in (
                    "ml/alphazero_lite/seed622_training_provenance_audit.py",
                    "ml/alphazero_lite/verify_seed622_training_provenance_audit.py",
                    "ml/alphazero_lite/seed599_training_executor.py",
                    "ml/alphazero_lite/verify_seed599_training_result.py",
                    "ml/alphazero_lite/seed615_training_executor.py",
                    "ml/alphazero_lite/verify_seed615_training_result.py",
                    "ml/alphazero_lite/verify_seed615_training_freeze.py",
                    "ml/alphazero_lite/verify_seed615_opening_diversity.py",
                    "ml/alphazero_lite/self_play.py",
                    "ml/alphazero_lite/input_encodings.py",
                    "ml/alphazero_lite/seed627_membership_snapshots.py",
                    "ml/alphazero_lite/verify_seed627_membership_snapshots.py",
                )
            },
        },
        "raw_input_sha256": _raw_input_hashes(),
        "source_verifier_results": verifier_results,
        "snapshot_counts": {key: len(value) for key, value in snapshots.items()},
        "snapshot_sha256": outputs,
        "snapshot_file_sha256": file_outputs,
        "training_lineage": lineage,
        "verification": verification,
        "provenance": {
            "source_game_opening": "authenticated_but_non_crosswalkable",
            "crosswalk_rows": [],
            "missing_or_insufficient": "Seed 624 opening_identity/game_id do not occur in historical B512/S615 membership ledgers; archive membership alone is not consumed-game proof",
            "inherited_PE4_ancestry": "unresolved_identity_incomplete_training_ledger",
        },
        "snapshot_schema": SCHEMA,
    }
    manifest_raw = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    with (output / "source-identity-lineage-manifest.json").open("xb") as stream:
        stream.write(manifest_raw)
    return manifest


def verify_snapshots(
    snapshots: dict[str, list[dict[str, Any]]], lineage: dict[str, Any]
) -> dict[str, Any]:
    """Validate identities, unique constraints, counts and lineage contract."""
    expected = {"B512": 2048, "S615": 2048, "protected_holdout": 1353}
    result = {}
    for name, count in expected.items():
        rows = snapshots[name]
        if len(rows) != count:
            raise ValueError(f"snapshot_count_mismatch:{name}")
        previous = b""
        canonical_ids: set[str] = set()
        float_ids: set[str] = set()
        for row in rows:
            state = row.get("canonical_state")
            if not isinstance(state, dict):
                raise ValueError(f"snapshot_state_missing:{name}")
            cid, fid = audit.identity(state), audit.float32_identity(state)
            if (cid, fid) != (
                row.get("canonical_identity"),
                row.get("production_float32_identity"),
            ):
                raise ValueError(f"snapshot_identity_mismatch:{name}")
            encoded = canonical(state)
            if encoded < previous:
                raise ValueError(f"snapshot_sort_order_mismatch:{name}")
            previous = encoded
            canonical_ids.add(cid)
            float_ids.add(fid)
        if name != "protected_holdout" and (
            len(canonical_ids) != count or len(float_ids) != count
        ):
            raise ValueError(f"snapshot_duplicate_identity:{name}")
        result[name] = {
            "rows": count,
            "unique_canonical": len(canonical_ids),
            "unique_float32": len(float_ids),
        }
    if set(lineage) != {"B512", "S615"}:
        raise ValueError("training_lineage_missing")
    return {"status": "verified", "populations": result}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.publish:
        raise SystemExit("use --publish for create-only snapshot publication")
    print(json.dumps(publish(args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
