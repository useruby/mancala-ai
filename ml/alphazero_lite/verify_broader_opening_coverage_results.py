"""Read-only verifier for the published broader-opening pilot results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ml.alphazero_lite import broader_opening_coverage_pilot as pilot
from ml.alphazero_lite import run_varied_opponent_pilot_v2 as v2
from ml.alphazero_lite.publish_broader_opening_coverage_results import (
    canonical_json,
    evidence_manifest,
    make_result,
)


FREEZE_SHA256 = "3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5"


def read_only_bind_run_directory(
    checkpoint_dir: Path, attempt_dir: Path, identity: dict[str, Any]
) -> Path:
    """Validate existing v2 binding and evidence without creating a marker."""
    marker = checkpoint_dir / "run-identity.v2"
    if not marker.is_file():
        raise ValueError("pilot_v2_run_identity_marker_missing")
    if json.loads(marker.read_text()) != identity:
        raise ValueError("pilot_v2_run_identity_conflict")
    for path in checkpoint_dir.glob("*.json"):
        payload = json.loads(path.read_text())
        if (
            payload.get("execution_identity") != identity
            or payload.get("result", {}).get("execution_identity") != identity
        ):
            raise ValueError("pilot_v2_checkpoint_identity_mismatch")
    for path in attempt_dir.glob("*.json"):
        if json.loads(path.read_text()).get("execution_identity") != identity:
            raise ValueError("pilot_v2_attempt_identity_mismatch")
    return marker


def verify(data: Path, expected_evidence_digest_sha256: str) -> dict[str, Any]:
    freeze = pilot.load_freeze(data / "freeze.json", FREEZE_SHA256)
    identity = {
        "schema": "broader_opening_pilot_v2_run_identity_v1",
        "freeze_sha256": freeze["freeze_sha256"],
        "runner_source": "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
        "runner_sha256": freeze["execution_source_hashes"][
            "ml/alphazero_lite/run_varied_opponent_pilot_v2.py"
        ],
        "probe_lifetime": v2.PAIR_LIFETIME,
        "checkpoint_schema": "varied_opponent_pair_checkpoint_v1",
    }
    # Fail before reading or analyzing incomplete data, and never delegate to the
    # runner's create-if-missing binder during verification.
    read_only_bind_run_directory(data / "pairs", data / "attempts", identity)
    published_result = json.loads((data / "results.json").read_text())
    published_digest = json.loads((data / "evidence-digest.json").read_text())
    if published_digest.get("digest_sha256") != expected_evidence_digest_sha256:
        raise ValueError("external_evidence_digest_mismatch")
    actual_digest = evidence_manifest(data)
    if published_digest != actual_digest:
        raise ValueError("raw_evidence_digest_mismatch")
    digest = published_digest.pop("digest_sha256")
    if hashlib.sha256(canonical_json(published_digest)).hexdigest() != digest:
        raise ValueError("evidence_digest_self_hash_mismatch")
    with patch.object(pilot.v2, "bind_run_directory", read_only_bind_run_directory):
        rebuilt_result, rebuilt_digest = make_result(data)
    if published_digest != {
        k: v for k, v in rebuilt_digest.items() if k != "digest_sha256"
    }:
        raise ValueError("evidence_manifest_reconstruction_mismatch")
    if published_result != rebuilt_result:
        raise ValueError("published_metrics_or_decision_mismatch")
    return {
        "passed": True,
        "freeze_sha256": published_result["freeze_sha256"],
        "evidence_digest_sha256": digest,
        "pairs_replayed_and_reconciled": rebuilt_result["analysis"]["pairs_reconciled"],
        "decision": rebuilt_result["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=pilot.DATA)
    parser.add_argument("--expected-evidence-digest-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.data, args.expected_evidence_digest_sha256),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
