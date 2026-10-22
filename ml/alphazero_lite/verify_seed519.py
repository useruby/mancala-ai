"""Read-only authentication supplement for the seed518 publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed515, verify_seed518_selective_head as seed518

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed518-selective-head"
REGISTRATION = ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linked(path: Path, expected: str, reason: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(reason)


def verify(directory: Path = PUBLICATION) -> dict[str, Any]:
    receipt_path = directory / "seed518-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    freeze_path = directory / "amended-timing-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    evidence_path = directory / "evidence.json"
    parity_path = directory / "parity-outputs.json"
    raw_path = directory / "raw-measurements.jsonl"
    amendment_path = directory / "comparison-contract-amendment.json"
    reanalysis_path = directory / "post-execution-reanalysis.json"
    identities = receipt["identities"]
    links = (
        (
            freeze_path,
            identities["amended_freeze_sha256"],
            "seed519_receipt_freeze_hash",
        ),
        (evidence_path, identities["evidence_sha256"], "seed519_receipt_evidence_hash"),
        (
            parity_path,
            identities["parity_archive_sha256"],
            "seed519_receipt_parity_hash",
        ),
        (raw_path, identities["raw_measurements_sha256"], "seed519_receipt_raw_hash"),
        (
            reanalysis_path,
            identities["post_execution_reanalysis_sha256"],
            "seed519_receipt_reanalysis_hash",
        ),
        (amendment_path, freeze["amendment_sha256"], "seed519_freeze_amendment_hash"),
    )
    for path, expected, reason in links:
        linked(path, expected, reason)

    evidence = json.loads(evidence_path.read_text())
    reanalysis = json.loads(reanalysis_path.read_text())
    if evidence.get("protocol_freeze_sha256") != digest(freeze_path):
        raise ValueError("seed519_evidence_freeze_cross_reference")
    if evidence.get("parity_sha256") != digest(parity_path):
        raise ValueError("seed519_evidence_parity_cross_reference")
    if evidence.get("post_execution_reanalysis_sha256") != digest(reanalysis_path):
        raise ValueError("seed519_evidence_reanalysis_cross_reference")
    if reanalysis.get("parity_archive_sha256") != digest(parity_path):
        raise ValueError("seed519_reanalysis_parity_cross_reference")
    if reanalysis.get("amendment") != amendment_path.name:
        raise ValueError("seed519_reanalysis_amendment_cross_reference")
    if digest(REGISTRATION) != freeze["registration_sha256"]:
        raise ValueError("seed519_registration_identity")
    registration = json.loads(REGISTRATION.read_text())
    for component in ("initializer", "E4"):
        artifact = ROOT / registration["components"][component]["artifact"]
        for filename in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            expected = freeze["components"][component][filename]
            if registration["components"][component]["files"][filename] != expected:
                raise ValueError(
                    f"seed519_registration_component:{component}:{filename}"
                )
            linked(
                artifact / filename,
                expected,
                f"seed519_component_bytes:{component}:{filename}",
            )

    for source, expected in freeze["source_sha256"].items():
        source_path = ROOT / source
        actual = digest(source_path)
        if source.endswith("verify_seed518_selective_head.py"):
            if receipt.get("verification_note", "").find(actual) < 0:
                raise ValueError("seed519_verifier_correction_identity")
            continue
        if actual != expected:
            raise ValueError(f"seed519_execution_source:{source}")

    # Authenticate runtime files with the seed517 pointer/materialization rule.
    registration_runtime = json.loads(REGISTRATION.read_text()).get(
        "runtime_assets", {}
    )
    # Seed518 freeze contains only paths and hashes; make equivalent runtime records
    # from the authoritative seed512 registration where available.
    for name, frozen in freeze["runtime_identities"].items():
        asset = registration_runtime.get(name)
        if asset is None:
            artifact = ROOT / frozen["path"].replace(str(ROOT) + "/", "")
            if not artifact.is_file() or digest(artifact) != frozen["sha256"]:
                raise ValueError(f"seed519_runtime_identity:{name}")
        else:
            verify_seed515.verify_runtime_asset(name, asset, frozen)

    semantic = seed518.verify(directory)
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    for row in raw_rows:
        if (
            row["dispatch_path"] == "exact_root"
            and row["puct_simulations_executed"] != 0
        ):
            raise ValueError("seed519_exact_root_simulation_accounting")
        if row["dispatch_path"] == "puct" and row["puct_simulations_executed"] != 1200:
            raise ValueError("seed519_puct_simulation_accounting")
    if semantic["decision"].get("advance") is not True:
        raise ValueError("seed519_fixed_decision")
    if receipt["original_gate"]["status"] != "failed-before-amendment":
        raise ValueError("seed519_historical_gate_record")
    return {
        "status": "verified",
        "publication": "seed518",
        "decision": semantic["decision"],
    }


def verify_supplement(directory: Path) -> dict[str, Any]:
    """Authenticate a separately published supplement and its create-only receipt."""
    supplement_path = directory / "supplement.json"
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if digest(supplement_path) != receipt.get("supplement_sha256"):
        raise ValueError("seed519_supplement_hash")
    if digest(Path(__file__)) != receipt.get("verifier_sha256"):
        raise ValueError("seed519_supplement_verifier_hash")
    if digest(REGISTRATION) != receipt.get("registration_sha256"):
        raise ValueError("seed519_supplement_registration_hash")
    if digest(PUBLICATION / "seed518-receipt.json") != receipt.get(
        "historical_seed518_receipt_sha256"
    ):
        raise ValueError("seed519_supplement_historical_receipt_hash")
    record = json.loads(supplement_path.read_text())
    if record.get("schema") != "seed519-seed518-authentication-supplement-v1":
        raise ValueError("seed519_supplement_schema")
    bindings = record.get("publication_sha256", {})
    for filename, key in (
        ("amended-timing-freeze.json", "amended_freeze_sha256"),
        ("evidence.json", "evidence_sha256"),
        ("raw-measurements.jsonl", "raw_measurements_sha256"),
        ("parity-outputs.json", "parity_sha256"),
        ("comparison-contract-amendment.json", "amendment_sha256"),
        ("post-execution-reanalysis.json", "reanalysis_sha256"),
    ):
        actual = digest(PUBLICATION / filename)
        if bindings.get(filename) != actual or receipt.get(key) != actual:
            raise ValueError(f"seed519_supplement_binding:{filename}")
    if record.get("registration_sha256") != receipt.get("registration_sha256"):
        raise ValueError("seed519_supplement_registration_binding")
    preservation_path = directory / "preservation.json"
    preservation_hash = (
        digest(preservation_path) if preservation_path.is_file() else None
    )
    if preservation_hash != receipt.get(
        "preservation_sha256"
    ) or preservation_hash != record.get("physical_copy_preservation_sha256"):
        raise ValueError("seed519_supplement_preservation_binding")
    if preservation_path.is_file():
        preservation = json.loads(preservation_path.read_text())
        if (
            preservation.get("status") != "verified"
            or preservation.get("unchanged_sha256_and_mtime_ns") is not True
            or preservation.get("historical_inventory_before")
            != preservation.get("historical_inventory_after")
        ):
            raise ValueError("seed519_supplement_preservation_invalid")
    if verify()["status"] != "verified":
        raise ValueError("seed519_supplement_seed518_verification")
    return {"status": "verified", "supplement": str(supplement_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=PUBLICATION)
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checked = (
        verify_supplement(args.directory) if args.supplement else verify(args.directory)
    )
    result = json.dumps(checked, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
