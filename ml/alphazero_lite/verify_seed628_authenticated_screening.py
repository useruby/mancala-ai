"""Independent reconstruction and screening verifier for Seed 628."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed624_prospective_benchmark as seed624_verifier
from ml.alphazero_lite import verify_seed627_membership_snapshots as seed627_verifier
from ml.alphazero_lite.seed628_authenticated_screening import (
    NAMES,
    PINNED_FILES,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def reconstruct_evidence(snapshot_dir: Path) -> dict[str, Any]:
    """Re-run Seed 627's pinned source reconstruction, then independently bind rows."""
    for filename, expected in PINNED_FILES.items():
        if _sha((snapshot_dir / filename).read_bytes()) != expected:
            raise ValueError(f"pinned_seed627_file_hash_mismatch:{filename}")
    recorded_numpy = seed627_verifier.np.__version__
    original_version = seed627_verifier.importlib.metadata.version
    original_run = seed627_verifier.subprocess.run
    historical_python = Path(__file__).resolve().parents[2] / ".venv/bin/python"

    def historical_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        return original_run([str(historical_python), *command[1:]], **kwargs)

    try:
        seed627_verifier.np.__version__ = "2.5.2"
        seed627_verifier.importlib.metadata.version = lambda package: (
            "2.13.0+cu130" if package == "torch" else original_version(package)
        )
        seed627_verifier.subprocess.run = historical_run
        verification = seed627_verifier.verify_bundle(snapshot_dir)
    finally:
        seed627_verifier.np.__version__ = recorded_numpy
        seed627_verifier.importlib.metadata.version = original_version
        seed627_verifier.subprocess.run = original_run

    manifest = json.loads(
        (snapshot_dir / "source-identity-lineage-manifest.json").read_bytes()
    )
    if (
        manifest.get("provenance", {}).get("source_game_opening")
        != "authenticated_but_non_crosswalkable"
        or manifest.get("provenance", {}).get("crosswalk_rows") != []
        or manifest.get("provenance", {}).get("inherited_PE4_ancestry")
        != "unresolved_identity_incomplete_training_ledger"
    ):
        raise ValueError("pinned_source_provenance_contract_mismatch")
    evidence: dict[str, Any] = {}
    for name in NAMES:
        raw = (snapshot_dir / f"{name}.json").read_bytes()
        rows = json.loads(raw)
        if raw != _canonical(rows) + b"\n":
            raise ValueError(f"snapshot_noncanonical_bytes:{name}")
        for row in rows:
            canonical_id, float_id = seed624_verifier._identities(
                row["canonical_state"]
            )
            if (canonical_id, float_id) != (
                row.get("canonical_identity"),
                row.get("production_float32_identity"),
            ):
                raise ValueError(f"cross_runtime_row_identity_mismatch:{name}")
        evidence[name] = {
            "authenticated": True,
            "rows": rows,
            "sha256": _sha(_canonical(rows)),
            "file_sha256": _sha(raw),
            "manifest_sha256": PINNED_FILES["source-identity-lineage-manifest.json"],
            "lineage_sha256": _sha(_canonical(manifest["training_lineage"])),
            "source_verification_sha256": _sha(
                _canonical(manifest["source_verifier_results"])
            ),
        }
    provenance = dict(manifest["provenance"])
    provenance.update(
        {
            "authenticated": True,
            "crosswalk_status": "authenticated_non_crosswalkable",
            "crosswalk_rows": [],
            "source_verification_status": "verified",
            "source_manifest_sha256": PINNED_FILES[
                "source-identity-lineage-manifest.json"
            ],
        }
    )
    provenance["sha256"] = _sha(
        _canonical(
            {key: value for key, value in provenance.items() if key != "authenticated"}
        )
    )
    evidence["source_provenance"] = provenance
    evidence["authentication"] = {
        "schema": "seed628-authenticated-seed627-adapter-v1",
        "seed627_verification": verification,
        "manifest_sha256": PINNED_FILES["source-identity-lineage-manifest.json"],
        "snapshot_file_sha256": {name: PINNED_FILES[f"{name}.json"] for name in NAMES},
        "identity_runtime": {
            "historical_numpy": "2.5.2",
            "screening_numpy": "2.5.3",
        },
    }
    return evidence


def reconstruct_screening(
    candidates: list[dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """Verifier-owned classification pass; does not call the Seed 624 producer."""
    inventories: dict[str, tuple[set[str], set[str]]] = {}
    for name in NAMES:
        source = evidence.get(name)
        if not isinstance(source, dict) or source.get("authenticated") is not True:
            raise ValueError(f"unauthenticated_membership:{name}")
        rows = source.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing_membership_rows:{name}")
        digest = _sha(_canonical(rows))
        if digest != source.get("sha256"):
            raise ValueError(f"membership_hash_mismatch:{name}")
        canonical_ids: set[str] = set()
        float_ids: set[str] = set()
        for row in rows:
            cid, fid = seed624_verifier._identities(row["canonical_state"])
            if cid != row.get("canonical_identity") or fid != row.get(
                "production_float32_identity"
            ):
                raise ValueError(f"membership_identity_mismatch:{name}")
            canonical_ids.add(cid)
            float_ids.add(fid)
        inventories[name] = canonical_ids, float_ids

    provenance = evidence.get("source_provenance", {})
    authentication = evidence.get("authentication", {})
    if (
        authentication.get("schema") != "seed628-authenticated-seed627-adapter-v1"
        or authentication.get("manifest_sha256")
        != PINNED_FILES["source-identity-lineage-manifest.json"]
        or authentication.get("identity_runtime")
        != {"historical_numpy": "2.5.2", "screening_numpy": "2.5.3"}
        or provenance.get("crosswalk_status") != "authenticated_non_crosswalkable"
        or provenance.get("crosswalk_rows") != []
    ):
        raise ValueError("seed628_authentication_or_crosswalk_claim_invalid")
    expected = provenance.get("sha256")
    payload = {
        key: value
        for key, value in provenance.items()
        if key not in {"authenticated", "sha256"}
    }
    if (
        provenance.get("authenticated") is not True
        or expected != _sha(_canonical(payload))
        or provenance.get("crosswalk_status")
        not in {"complete", "authenticated_non_crosswalkable"}
    ):
        raise ValueError("source_provenance_missing_or_unauthenticated")
    crosswalkable = provenance["crosswalk_status"] == "complete"
    if crosswalkable and any(
        not isinstance(provenance.get(name), list) for name in NAMES
    ):
        raise ValueError("source_provenance_crosswalk_incomplete")

    result = []
    for candidate in candidates:
        cid = candidate["canonical_identity"]
        fid = candidate["production_float32_identity"]
        hits = {
            name: cid in canonical_ids or fid in float_ids
            for name, (canonical_ids, float_ids) in inventories.items()
        }
        linked: bool | None = False if crosswalkable else None
        if crosswalkable:
            linked = any(
                entry.get("opening_identity") == candidate["opening_identity"]
                or entry.get("game_id") == candidate["game_id"]
                for name in NAMES
                for entry in provenance[name]
            )
        classification = (
            "confirmed_known_exposure"
            if any(hits.values()) or linked
            else "known_exposure_nonmatch_inherited_PE4_unresolved"
        )
        result.append(
            {
                **candidate,
                "exposure": hits,
                "source_provenance_match": linked,
                "source_provenance_status": provenance["crosswalk_status"],
                "classification": classification,
                "membership_evidence_sha256": {
                    name: evidence[name]["sha256"] for name in NAMES
                },
                "inherited_PE4_exposure": "unresolved",
            }
        )
    return result


def verify_screening(
    candidates: list[dict[str, Any]],
    evidence: dict[str, Any],
    published: list[dict[str, Any]],
) -> dict[str, Any]:
    reconstructed = reconstruct_screening(candidates, evidence)
    if _canonical(reconstructed) != _canonical(published):
        raise ValueError("independent_screening_reconstruction_mismatch")
    return {
        "status": "verified",
        "classifications": {
            label: sum(row["classification"] == label for row in reconstructed)
            for label in (
                "confirmed_known_exposure",
                "known_exposure_nonmatch_inherited_PE4_unresolved",
            )
        },
        "source_provenance_status": evidence["source_provenance"]["crosswalk_status"],
        "inherited_PE4_clearance_claimed": False,
    }
