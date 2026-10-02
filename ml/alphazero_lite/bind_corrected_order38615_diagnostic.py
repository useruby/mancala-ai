"""Bind immutable artifacts and the pre-registered corrected suite bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.alphazero_lite.build_opening_suite import (
    load_suite_jsonl,
    validate_arena_entries,
)
from ml.alphazero_lite.opening_exclusion_contract import (
    validate_suite_against_manifest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/order38615-a5-frozen-diagnostic-v4"
REG = DATA / "registration.json"
BIND = DATA / "evaluation-binding.json"
SOURCE_BIND = ROOT / "docs/data/order38615-a5-confirmation-candidate-binding.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind() -> None:
    registration = json.loads(REG.read_text())
    source = json.loads(SOURCE_BIND.read_text())
    candidate = registration["candidate"]
    manifest_path = ROOT / registration["exclusion_proof"]["manifest_path"]
    if sha(manifest_path) != registration["exclusion_proof"]["manifest_sha256"]:
        raise ValueError("exclusion_manifest_hash_mismatch")
    manifest = verify_manifest(manifest_path)
    if source["candidate"]["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
        raise ValueError("candidate_checkpoint_binding_mismatch")
    if source["candidate"]["artifact"] != candidate["artifact"]:
        raise ValueError("candidate_artifact_binding_mismatch")
    for filename, digest in candidate["artifact_sha256"].items():
        if sha(Path(candidate["artifact"]) / filename) != digest:
            raise ValueError(f"candidate_artifact_hash_mismatch:{filename}")
    all_identities: set[str] = set()
    for seed, spec in registration["evaluation"]["suites"].items():
        suite_path = ROOT / spec["path"]
        if sha(suite_path) != spec["sha256"]:
            raise ValueError(f"suite_hash_mismatch:{seed}")
        rows = load_suite_jsonl(str(suite_path))
        identities = validate_arena_entries(rows)
        if len(rows) != 512 or len(set(identities)) != 512:
            raise ValueError(f"suite_identity_mismatch:{seed}")
        excluded = validate_suite_against_manifest(rows, manifest)
        if excluded != set(identities) or excluded & all_identities:
            raise ValueError(f"suite_exclusion_or_cross_overlap:{seed}")
        all_identities |= excluded
    value = {
        "schema": "order38615-corrected-diagnostic-evaluation-binding-v1",
        "registration_sha256": sha(REG),
        "exclusion_manifest_sha256": sha(manifest_path),
        "source_candidate_binding_sha256": sha(SOURCE_BIND),
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "opponent": registration["opponent"],
        "suite_sha256": {
            seed: spec["sha256"]
            for seed, spec in registration["evaluation"]["suites"].items()
        },
        "status": "bound_before_games",
        "reports": {},
    }
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if BIND.exists() and BIND.read_text() != payload:
        raise ValueError("immutable_binding_conflict")
    if not BIND.exists():
        BIND.write_text(payload)
    print(f"binding_sha256={sha(BIND)}")


if __name__ == "__main__":
    bind()
