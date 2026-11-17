"""Create the post-verification receipt after the read-only verifier passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed604_opening_cohort import PUBLICATION, ROOT
from ml.alphazero_lite.verify_seed604_opening_cohort import verify


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def publish(root: Path = ROOT) -> dict[str, Any]:
    """Verify first, then create a receipt exactly once without changing evidence."""
    result = verify(root)
    folder = root / PUBLICATION
    registration_raw = (folder / "registration.json").read_bytes()
    registration = json.loads(registration_raw)
    source_raw = Path(__file__).read_bytes()
    receipt = {
        "schema": "seed604-opening-cohort-verification-v1",
        "status": "passed",
        "verification": result,
        "registration_sha256": sha256(registration_raw),
        "manifest_sha256": registration["manifest_sha256"],
        "verifier_source_sha256": registration["verifier_source_sha256"],
        "receipt_publisher_source_sha256": sha256(source_raw),
        "limitations": [
            "Seed 461 continuation identities remain unresolved and excluded.",
            "Other out-of-scope historical populations have unknown counts.",
            "Seed 599 post-root trajectories are not included; only its registered roots are excluded.",
        ],
    }
    payload = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
    report_path = folder / "verification-report.json"
    with report_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return {"passed": True, "verification_report_sha256": sha256(payload), **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(publish(), sort_keys=True, indent=2))
