"""Independent verifier for physically copied Seed 627 snapshot evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_NAMES = ("B512", "S615", "protected_holdout")
S599 = Path("docs/data/seed599-replay-coverage")
S615 = Path("docs/data/seed615-source-opening-diversity")
SOURCE_IDENTITY_PATHS = (
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
EXPECTED = {
    "B512": "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
    "S615": "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010",
    "protected_holdout": "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8",
    "B512_trace": "9f22328080756aa0fa2fa066164d36f935643938363059e0d0c711eb6f8f8955",
    "S615_trace": "8ee2252f6831ea0589a19dff2c1ebc79a0f3c3447764feb9c53086ba599a8b01",
    "parent": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "B512_freeze": "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9",
    "S615_freeze": "6ac9d4b4e52a6f4789ce406d9c88badd82e9c8ab7cf4b4d4d8e9c8676cd9910d",
    "B512_checkpoint": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615_checkpoint": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def ids(state: dict[str, Any]) -> tuple[str, str]:
    cid = sha(canonical(state))
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return cid, sha(vector.tobytes())


def _membership(name: str) -> tuple[list[dict[str, Any]], str]:
    if name == "B512":
        raw = (ROOT / S599 / "treatment-membership.json").read_bytes()
        parsed = json.loads(raw)
    elif name == "S615":
        raw = (ROOT / S615 / "candidate-membership.jsonl").read_bytes()
        parsed = [json.loads(line) for line in raw.splitlines() if line]
    else:
        raw = (ROOT / S599 / "heldout-membership.json").read_bytes()
        parsed = json.loads(raw)
    if sha(canonical(parsed)) != EXPECTED[name]:
        raise ValueError(f"historical_membership_digest_mismatch:{name}")
    return parsed, sha(raw)


def reconstruct_consumed(
    name: str, members: list[dict[str, Any]]
) -> tuple[list[int], dict]:
    directory = ROOT / (
        S599 / "training-run-B" if name == "B512" else S615 / "training-run-S615"
    )
    trace_raw = (directory / "training-trace.jsonl").read_bytes()
    if sha(trace_raw) != EXPECTED[f"{name}_trace"]:
        raise ValueError(f"historical_trace_digest_mismatch:{name}")
    schedule = json.loads((ROOT / S599 / "batch-schedule-B.json").read_bytes())
    trace = [json.loads(line) for line in trace_raw.splitlines() if line]
    if len(schedule) != 512 or len(trace) != 512:
        raise ValueError(f"historical_update_count_mismatch:{name}")
    used: set[int] = set()
    for update, (record, batch) in enumerate(zip(trace, schedule, strict=True), 1):
        if record.get("update") != update or record.get("indexes") != batch:
            raise ValueError(f"trace_schedule_mismatch:{name}:{update}")
        if len(batch) != 32 or any(
            type(index) is not int or not 0 <= index < len(members) for index in batch
        ):
            raise ValueError(f"invalid_consumed_index:{name}:{update}")
        used.update(batch)
    if used != set(range(len(members))):
        raise ValueError(f"incomplete_consumption:{name}")
    receipt = json.loads((directory / "training-receipt.json").read_bytes())
    if receipt.get("trace_sha256") != sha(trace_raw) or receipt.get("updates") != 512:
        raise ValueError(f"receipt_trace_binding_mismatch:{name}")
    freeze_path = ROOT / (
        S599 / "training-execution-freeze.json"
        if name == "B512"
        else S615 / "training-execution-freeze-v3.json"
    )
    freeze_raw = freeze_path.read_bytes()
    if sha(freeze_raw) != EXPECTED[f"{name}_freeze"]:
        raise ValueError(f"training_freeze_hash_mismatch:{name}")
    membership_hash = EXPECTED[name]
    if (
        receipt.get("membership_sha256") != membership_hash
        or receipt.get("parent_checkpoint_sha256") != EXPECTED["parent"]
        or receipt.get("batch_schedule_sha256") != sha(canonical(schedule))
        or receipt.get("execution_freeze_sha256") != sha(freeze_raw)
        or receipt.get("checkpoint_sha256") != EXPECTED[f"{name}_checkpoint"]
    ):
        raise ValueError(f"training_receipt_binding_mismatch:{name}")
    return sorted(used), {
        "updates": 512,
        "batch_size": 32,
        "unique_membership_rows_consumed": len(used),
        "membership_rows": len(members),
        "membership_sha256": membership_hash,
        "batch_schedule_sha256": sha(canonical(schedule)),
        "execution_freeze_sha256": sha(freeze_raw),
        "parent_checkpoint_sha256": EXPECTED["parent"],
        "checkpoint_sha256": EXPECTED[f"{name}_checkpoint"],
        "trace_sha256": sha(trace_raw),
        "receipt_sha256": sha((directory / "training-receipt.json").read_bytes()),
    }


def reconstruct() -> tuple[dict[str, list[dict]], dict[str, Any], dict[str, str]]:
    # This established Seed 622 verifier independently checks all frozen source
    # hashes, receipts, schedules, parent and checkpoint ancestry, and prior audit.
    from ml.alphazero_lite.verify_seed622_training_provenance_audit import verify

    verify()
    members = {name: _membership(name)[0] for name in SNAPSHOT_NAMES}
    snapshot_sources: dict[str, list[tuple[int, dict]]] = {}
    lineage = {}
    for name in ("B512", "S615"):
        consumed, lineage[name] = reconstruct_consumed(name, members[name])
        snapshot_sources[name] = [(index, members[name][index]) for index in consumed]
    snapshot_sources["protected_holdout"] = list(
        enumerate(members["protected_holdout"])
    )
    output: dict[str, list[dict]] = {}
    for name, indexed_rows in snapshot_sources.items():
        rows = []
        for index, source in indexed_rows:
            state = source.get("canonical_state")
            if not isinstance(state, dict):
                raise ValueError(f"source_state_invalid:{name}:{index}")
            cid, fid = ids(state)
            if source.get("canonical_identity") != cid:
                raise ValueError(f"canonical_identity_mismatch:{name}:{index}")
            declared_fid = source.get("float32_identity")
            if declared_fid is not None and declared_fid != fid:
                raise ValueError(f"float32_identity_mismatch:{name}:{index}")
            rows.append(
                {
                    "canonical_state": state,
                    "canonical_identity": cid,
                    "production_float32_identity": fid,
                    "source_membership_index": index,
                    "lineage_evidence": {
                        "population": name,
                        "membership_source": (
                            str(S599 / "treatment-membership.json")
                            if name == "B512"
                            else str(S615 / "candidate-membership.jsonl")
                            if name == "S615"
                            else str(S599 / "heldout-membership.json")
                        ),
                        "consumed_by_training": name in {"B512", "S615"},
                        "training_updates": 512 if name in {"B512", "S615"} else None,
                        "registered_batch_size": 32
                        if name in {"B512", "S615"}
                        else None,
                    },
                }
            )
        rows.sort(key=lambda item: canonical(item["canonical_state"]))
        output[name] = rows
    input_hashes = {name: _membership(name)[1] for name in SNAPSHOT_NAMES}
    return output, lineage, input_hashes


def raw_input_hashes() -> dict[str, str]:
    paths = {
        "B512_membership": S599 / "treatment-membership.json",
        "B512_schedule": S599 / "batch-schedule-B.json",
        "B512_trace": S599 / "training-run-B/training-trace.jsonl",
        "B512_receipt": S599 / "training-run-B/training-receipt.json",
        "B512_freeze": S599 / "training-execution-freeze.json",
        "B512_independent_verification": S599
        / "training-run-B/independent-verification.json",
        "S615_membership": S615 / "candidate-membership.jsonl",
        "S615_trace": S615 / "training-run-S615/training-trace.jsonl",
        "S615_receipt": S615 / "training-run-S615/training-receipt.json",
        "S615_freeze": S615 / "training-execution-freeze-v3.json",
        "S615_integrity_report": S615
        / "training-run-S615/training-integrity-report.json",
        "S615_runtime_certification": S615 / "training-runtime-certification-v3.json",
        "S615_independent_verification": S615 / "independent-verification.json",
        "protected_holdout": S599 / "heldout-membership.json",
        "shared_parent_checkpoint": Path(
            "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz"
        ),
        "seed599_preparation": S599 / "preparation.json",
    }
    return {key: sha((ROOT / path).read_bytes()) for key, path in paths.items()}


def rerun_historical_verifiers() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed599_training_result",
            "--arm",
            "B",
            "--output-dir",
            str(S599 / "training-run-B"),
        ],
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed615_training_result",
            "--output-dir",
            str(S615 / "training-run-S615"),
        ],
        [
            sys.executable,
            "-c",
            "from ml.alphazero_lite.verify_seed615_training_freeze import verify; import json; print(json.dumps(verify(allow_execution_artifacts=True)))",
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
        run = subprocess.run(
            command, cwd=ROOT, capture_output=True, check=True, text=True
        )
        results[command[2] if len(command) > 2 else command[1]] = json.loads(run.stdout)
    return results


def validate_provenance(provenance: dict[str, Any]) -> None:
    if (
        provenance.get("source_game_opening") != "authenticated_but_non_crosswalkable"
        or provenance.get("crosswalk_rows") != []
        or not provenance.get("missing_or_insufficient")
        or provenance.get("inherited_PE4_ancestry")
        != "unresolved_identity_incomplete_training_ledger"
    ):
        raise ValueError("source_provenance_crosswalk_or_parent_claim_invalid")


def validate_dependency_identity(manifest: dict[str, Any]) -> None:
    identity = manifest.get("dependency_identity", {})
    lock = "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
    if (
        identity.get("runtime_lock") != lock
        or identity.get("runtime_lock_sha256") != sha((ROOT / lock).read_bytes())
        or identity.get("source_sha256")
        != {path: sha((ROOT / path).read_bytes()) for path in SOURCE_IDENTITY_PATHS}
    ):
        raise ValueError("dependency_or_source_identity_mismatch")
    runtime = manifest.get("runtime", {})
    if runtime != {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": importlib.metadata.version("torch"),
    }:
        raise ValueError("historical_verifier_runtime_identity_mismatch")


def validate_snapshot_payload(
    name: str,
    raw: bytes,
    expected_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = json.loads(raw)
    if rows != expected_rows or raw != canonical(rows) + b"\n":
        raise ValueError(f"snapshot_rows_omitted_added_or_modified:{name}")
    if sha(canonical(rows)) != manifest.get("snapshot_sha256", {}).get(name) or sha(
        raw
    ) != manifest.get("snapshot_file_sha256", {}).get(name):
        raise ValueError(f"snapshot_digest_mismatch:{name}")
    return rows


def verify_bundle(bundle_dir: Path, *, source_root: Path = ROOT) -> dict[str, Any]:
    global ROOT
    ROOT = source_root.resolve()
    expected_rows, lineage, input_hashes = reconstruct()
    manifest_raw = (bundle_dir / "source-identity-lineage-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != "seed627-source-identity-lineage-manifest-v1":
        raise ValueError("manifest_schema_mismatch")
    if manifest.get("status") != "authenticated":
        raise ValueError("snapshot_manifest_not_authenticated")
    if manifest.get("historical_evidence_modified") is not False:
        raise ValueError("historical_evidence_modification_claim_invalid")
    validate_dependency_identity(manifest)
    if manifest.get("raw_input_sha256") != raw_input_hashes():
        raise ValueError("manifest_source_input_hash_mismatch")
    if manifest.get("training_lineage") != lineage:
        raise ValueError("manifest_lineage_mismatch")
    if manifest.get("source_verifier_results") != rerun_historical_verifiers():
        raise ValueError("historical_verifier_results_mismatch")
    counts = {}
    for name, expected in expected_rows.items():
        raw = (bundle_dir / f"{name}.json").read_bytes()
        rows = validate_snapshot_payload(name, raw, expected, manifest)
        canonical_ids = [row["canonical_identity"] for row in rows]
        float_ids = [row["production_float32_identity"] for row in rows]
        if name != "protected_holdout" and (
            len(set(canonical_ids)) != len(rows) or len(set(float_ids)) != len(rows)
        ):
            raise ValueError(f"duplicate_identity:{name}")
        counts[name] = len(rows)
    if manifest.get("snapshot_counts") != counts:
        raise ValueError("snapshot_aggregate_count_mismatch")
    validate_provenance(manifest.get("provenance", {}))
    return {
        "status": "verified",
        "counts": counts,
        "independent_source_reconstruction": True,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_bundle(args.bundle, source_root=args.source_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
