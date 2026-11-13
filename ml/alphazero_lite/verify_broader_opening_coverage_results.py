"""Read-only verifier for the published broader-opening pilot results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import broader_opening_coverage_pilot as pilot
from ml.alphazero_lite.publish_broader_opening_coverage_results import (
    canonical_json,
    evidence_manifest,
    make_result,
)


def verify(data: Path) -> dict[str, Any]:
    published_result = json.loads((data / "results.json").read_text())
    published_digest = json.loads((data / "evidence-digest.json").read_text())
    actual_digest = evidence_manifest(data)
    if published_digest != actual_digest:
        raise ValueError("raw_evidence_digest_mismatch")
    digest = published_digest.pop("digest_sha256")
    if hashlib.sha256(canonical_json(published_digest)).hexdigest() != digest:
        raise ValueError("evidence_digest_self_hash_mismatch")
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
    args = parser.parse_args()
    print(json.dumps(verify(args.data), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
