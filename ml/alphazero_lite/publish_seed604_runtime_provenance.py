"""Create a create-only runtime/dependency identity receipt for Seed 604."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed604_opening_cohort import PUBLICATION, ROOT
from ml.alphazero_lite.verify_seed604_opening_cohort import verify


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def publish(root: Path = ROOT) -> dict[str, Any]:
    verification = verify(root)
    folder = root / PUBLICATION
    registration_raw = (folder / "registration.json").read_bytes()
    registration = json.loads(registration_raw)
    report_raw = (folder / "verification-report.json").read_bytes()
    random_source = Path(random.__file__).resolve()
    evidence = {
        "schema": "seed604-opening-cohort-runtime-provenance-v1",
        "registration_sha256": sha256(registration_raw),
        "manifest_sha256": registration["manifest_sha256"],
        "generator_source_commit": registration["source_commit"],
        "generator_source_sha256": registration["generator_source_sha256"],
        "verifier_source_sha256": registration["verifier_source_sha256"],
        "rules_source_sha256": registration["rules_source_sha256"],
        "identity_encoder_source_sha256": registration["identity_source_sha256"],
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        "python_implementation_cache_tag": sys.implementation.cache_tag,
        "python_random_module_sha256": sha256(random_source.read_bytes()),
        "python_random_module_path": random_source.name,
        "verification_report_sha256": sha256(report_raw),
        "verification_passed": verification["passed"],
    }
    payload = json.dumps(evidence, sort_keys=True, indent=2).encode() + b"\n"
    destination = folder / "runtime-provenance.json"
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return {"passed": True, "runtime_provenance_sha256": sha256(payload), **evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(publish(), sort_keys=True, indent=2))
