"""Independent reconstruction and screening verifier for Seed 628."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed624_prospective_benchmark as seed624_verifier

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "docs/data/seed627-authenticated-membership/snapshots"
NAMES = ("B512", "S615", "protected_holdout")
PINNED_FILES = {
    "B512.json": "de7650a5d4f0d7a9dec11adb3d100689381a4b1539ec0c9749a0f82feda28d59",
    "S615.json": "8ed6f059d491fa4a6bfd90c942d6e02d05ef1b6a82db65969402ad4351a084fd",
    "protected_holdout.json": "4716976a654ecba4aadccfa336d0dc3c5fdbd354ac27786848a06485d6062610",
    "source-identity-lineage-manifest.json": "e7a6af69f72780ca5ca79f9fe8214b9d347b6c6c0fb7cfea87b0617e3a16d3f3",
}
HISTORICAL_PYTHON = ROOT / ".venv/bin/python"
HISTORICAL_RUNTIME_LOCK = (
    ROOT
    / "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
)
HISTORICAL_RUNTIME_LOCK_SHA256 = (
    "5762f889f9719525379f8f1321811df9c52ea3675cdb319492eba15570c94ce4"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _runtime_probe() -> dict[str, Any]:
    code = (
        "import importlib.metadata, json, sys; "
        "from pathlib import Path; import re; "
        "lock=Path(sys.argv[1]); "
        "names=[re.match(r'^([A-Za-z0-9_.-]+)==([^ ]+)', line) "
        "for line in lock.read_text().splitlines()]; "
        "names=[m.group(1) for m in names if m]; "
        "packages={name.lower().replace('_','-'): "
        "importlib.metadata.version(name) for name in names}; "
        "import numpy, torch; "
        "print(json.dumps({'python':sys.version,'numpy':numpy.__version__,"
        "'torch':torch.__version__,'packages':packages},sort_keys=True))"
    )
    run = subprocess.run(
        [str(HISTORICAL_PYTHON), "-c", code, str(HISTORICAL_RUNTIME_LOCK)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(run.stdout)
    runtime["probe_stdout_sha256"] = _sha(run.stdout.encode("utf-8"))
    return runtime


def _locked_packages() -> dict[str, str]:
    raw = HISTORICAL_RUNTIME_LOCK.read_bytes()
    if _sha(raw) != HISTORICAL_RUNTIME_LOCK_SHA256:
        raise ValueError("historical_runtime_lock_hash_mismatch")
    packages = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match:
            packages[match.group(1).lower().replace("_", "-")] = match.group(2)
    return packages


def _proposed_runtime_probe() -> dict[str, str]:
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys,numpy; "
            "print(json.dumps({'python':sys.version,'numpy':numpy.__version__},sort_keys=True))",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(run.stdout)
    runtime["probe_stdout_sha256"] = _sha(run.stdout.encode("utf-8"))
    return runtime


def reconstruct_evidence(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    """Re-run historical source verification and independently bind pinned rows."""
    for filename, expected in PINNED_FILES.items():
        if _sha((snapshot_dir / filename).read_bytes()) != expected:
            raise ValueError(f"pinned_seed627_file_hash_mismatch:{filename}")
    manifest = json.loads(
        (snapshot_dir / "source-identity-lineage-manifest.json").read_bytes()
    )
    runtime = _runtime_probe()
    locked = _locked_packages()
    if (
        runtime.get("packages") != locked
        or runtime.get("python") != manifest.get("runtime", {}).get("python")
        or runtime.get("numpy") != manifest.get("runtime", {}).get("numpy")
        or runtime.get("torch") != manifest.get("runtime", {}).get("torch")
        or runtime.get("numpy") != "2.5.2"
    ):
        raise ValueError("historical_runtime_does_not_match_pinned_evidence")
    proposed_runtime = _proposed_runtime_probe()
    if (
        proposed_runtime.get("numpy") != "2.5.3"
        or proposed_runtime.get("numpy") != seed624_verifier.np.__version__
    ):
        raise ValueError("proposed_seed624_numpy_runtime_mismatch")
    verifier_command = [
        str(HISTORICAL_PYTHON),
        "-m",
        "ml.alphazero_lite.verify_seed627_membership_snapshots",
        str(snapshot_dir.resolve()),
        "--source-root",
        str(ROOT),
    ]
    verification_run = subprocess.run(
        verifier_command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    verification_result = json.loads(verification_run.stdout)
    if verification_result.get("status") != "verified":
        raise ValueError("historical_seed627_verifier_failed")
    verification = {
        "result": verification_result,
        "stdout_sha256": _sha(verification_run.stdout.encode("utf-8")),
        "stderr_sha256": _sha(verification_run.stderr.encode("utf-8")),
        "command": verifier_command,
    }
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
        "historical_runtime_lock_sha256": HISTORICAL_RUNTIME_LOCK_SHA256,
        "historical_runtime": runtime,
        "seed627_verification": verification,
        "manifest_sha256": PINNED_FILES["source-identity-lineage-manifest.json"],
        "snapshot_file_sha256": {name: PINNED_FILES[f"{name}.json"] for name in NAMES},
        "identity_runtime": {
            "historical_numpy": runtime["numpy"],
            "screening_numpy": proposed_runtime["numpy"],
            "screening_python": proposed_runtime["python"],
            "screening_probe_stdout_sha256": proposed_runtime["probe_stdout_sha256"],
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
        != {
            "historical_numpy": "2.5.2",
            "screening_numpy": "2.5.3",
            "screening_python": sys.version,
            "screening_probe_stdout_sha256": _proposed_runtime_probe()[
                "probe_stdout_sha256"
            ],
        }
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


def require_pinned_evidence(
    submitted: dict[str, Any], reconstructed: dict[str, Any]
) -> None:
    """Reject evidence that is not byte-for-byte the reconstructed pinned object."""
    if _canonical(submitted) != _canonical(reconstructed):
        raise ValueError("production_evidence_does_not_match_pinned_seed627_chain")


def require_screening_matches(
    reconstructed: list[dict[str, Any]], published: list[dict[str, Any]]
) -> None:
    """Require the published per-candidate labels to match independent results."""
    if _canonical(reconstructed) != _canonical(published):
        raise ValueError("independent_screening_reconstruction_mismatch")


def verify_screening(
    candidates: list[dict[str, Any]],
    evidence: dict[str, Any],
    published: list[dict[str, Any]],
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> dict[str, Any]:
    authentic = reconstruct_evidence(snapshot_dir)
    require_pinned_evidence(evidence, authentic)
    reconstructed = reconstruct_screening(candidates, evidence)
    require_screening_matches(reconstructed, published)
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
