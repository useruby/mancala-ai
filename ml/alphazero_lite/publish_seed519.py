"""Create a separate seed519 verification supplement and receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from ml.alphazero_lite import verify_seed519


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(output: Path) -> None:
    report = verify_seed519.verify()
    publication = verify_seed519.PUBLICATION
    freeze = json.loads((publication / "amended-timing-freeze.json").read_text())
    registration = json.loads(verify_seed519.REGISTRATION.read_text())
    component_bytes = {
        component: {
            filename: sha(
                verify_seed519.ROOT
                / registration["components"][component]["artifact"]
                / filename
            )
            for filename in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        for component in ("initializer", "E4")
    }
    publication_bindings = {
        filename: sha(publication / filename)
        for filename in (
            "seed518-receipt.json",
            "amended-timing-freeze.json",
            "evidence.json",
            "raw-measurements.jsonl",
            "parity-outputs.json",
            "comparison-contract-amendment.json",
            "post-execution-reanalysis.json",
        )
    }
    preservation = output / "preservation.json"
    supplement = {
        "schema": "seed519-seed518-authentication-supplement-v1",
        "publication": "docs/data/seed518-selective-head",
        "verification": report,
        "establishes": "present-byte authentication of published evidence, components, registered identities, execution sources, and runtime assets",
        "historical_launch_check_omission": "seed518 run_timing() did not perform component-file checks; this supplement authenticates current bytes against the amended freeze and authoritative registration, and does not establish that the historical launch performed those checks",
        "verifier_correction": "the current verifier source identity is checked against seed518-receipt.json; the pre-measurement frozen verifier hash is retained as historical identity and is not asserted to equal the post-timing corrected source",
        "scope": "read-only authentication; no searches, timing runs, games, training, deployment, promotion, or artifact replacement",
        "publication_sha256": publication_bindings,
        "registration_sha256": sha(verify_seed519.REGISTRATION),
        "component_sha256": component_bytes,
        "execution_source_sha256": {
            source: sha(verify_seed519.ROOT / source)
            for source in freeze["source_sha256"]
        },
        "runtime_sha256": {
            name: item["sha256"] for name, item in freeze["runtime_identities"].items()
        },
        "verification_environment": {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "verifier_path": str(Path(verify_seed519.__file__).resolve()),
            "dependency_paths": {
                "verify_seed515": str(
                    Path(verify_seed519.verify_seed515.__file__).resolve()
                ),
                "verify_seed518": str(Path(verify_seed519.seed518.__file__).resolve()),
            },
        },
        "physical_copy_preservation_sha256": sha(preservation)
        if preservation.is_file()
        else None,
    }
    encoded = json.dumps(supplement, indent=2, sort_keys=True) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    supplement_path = output / "supplement.json"
    with supplement_path.open("x") as stream:
        stream.write(encoded)
    receipt = {
        "schema": "seed519-supplement-receipt-v1",
        "supplement_sha256": sha(supplement_path),
        "verifier_sha256": sha(Path(verify_seed519.__file__)),
        "historical_seed518_receipt_sha256": sha(
            verify_seed519.PUBLICATION / "seed518-receipt.json"
        ),
        "registration_sha256": sha(verify_seed519.REGISTRATION),
        "amended_freeze_sha256": publication_bindings["amended-timing-freeze.json"],
        "evidence_sha256": publication_bindings["evidence.json"],
        "raw_measurements_sha256": publication_bindings["raw-measurements.jsonl"],
        "parity_sha256": publication_bindings["parity-outputs.json"],
        "amendment_sha256": publication_bindings["comparison-contract-amendment.json"],
        "reanalysis_sha256": publication_bindings["post-execution-reanalysis.json"],
        "preservation_sha256": sha(preservation) if preservation.is_file() else None,
        "result": "verified",
    }
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output)


if __name__ == "__main__":
    main()
