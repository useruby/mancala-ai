"""Read-only bridge from authenticated Seed 627 membership to Seed 624 screening."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed624_prospective_benchmark as seed624

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "docs/data/seed627-authenticated-membership/snapshots"
PINNED_FILES = {
    "B512.json": "de7650a5d4f0d7a9dec11adb3d100689381a4b1539ec0c9749a0f82feda28d59",
    "S615.json": "8ed6f059d491fa4a6bfd90c942d6e02d05ef1b6a82db65969402ad4351a084fd",
    "protected_holdout.json": "4716976a654ecba4aadccfa336d0dc3c5fdbd354ac27786848a06485d6062610",
    "source-identity-lineage-manifest.json": "e7a6af69f72780ca5ca79f9fe8214b9d347b6c6c0fb7cfea87b0617e3a16d3f3",
}
NAMES = ("B512", "S615", "protected_holdout")
HISTORICAL_PYTHON = ROOT / ".venv/bin/python"
HISTORICAL_RUNTIME_LOCK = (
    ROOT
    / "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
)
HISTORICAL_RUNTIME_LOCK_SHA256 = (
    "5762f889f9719525379f8f1321811df9c52ea3675cdb319492eba15570c94ce4"
)
HISTORICAL_VERIFIER_MODULE = "ml.alphazero_lite.verify_seed627_membership_snapshots"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _locked_packages() -> dict[str, str]:
    lock_raw = HISTORICAL_RUNTIME_LOCK.read_bytes()
    if sha256(lock_raw) != HISTORICAL_RUNTIME_LOCK_SHA256:
        raise ValueError("historical_runtime_lock_hash_mismatch")
    packages = {}
    for line in lock_raw.decode("utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match:
            packages[match.group(1).lower().replace("_", "-")] = match.group(2)
    if not packages:
        raise ValueError("historical_runtime_lock_empty")
    return packages


def _historical_runtime_probe(
    python: Path = HISTORICAL_PYTHON,
) -> dict[str, Any]:
    """Probe the genuine historical interpreter and all lock-pinned packages."""
    if not python.is_file():
        raise FileNotFoundError("pinned_historical_python_missing")
    code = (
        "import importlib.metadata, json, sys; "
        "from pathlib import Path; "
        "lock=Path(sys.argv[1]); "
        "import re; "
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
        [str(python), "-c", code, str(HISTORICAL_RUNTIME_LOCK)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(run.stdout)
    runtime["probe_stdout_sha256"] = sha256(run.stdout.encode("utf-8"))
    return runtime


def _validate_historical_runtime(
    runtime: dict[str, Any], manifest: dict[str, Any]
) -> None:
    locked = _locked_packages()
    if runtime.get("packages") != locked:
        raise ValueError("historical_runtime_packages_do_not_match_lock")
    recorded = manifest.get("runtime", {})
    if (
        runtime.get("python") != recorded.get("python")
        or runtime.get("numpy") != recorded.get("numpy")
        or runtime.get("torch") != recorded.get("torch")
        or runtime.get("numpy") != "2.5.2"
    ):
        raise ValueError("historical_runtime_does_not_match_pinned_evidence")


def _run_historical_seed627_verifier(snapshot_dir: Path) -> dict[str, Any]:
    """Execute the unchanged Seed 627 verifier under the pinned historical Python."""
    command = [
        str(HISTORICAL_PYTHON),
        "-m",
        HISTORICAL_VERIFIER_MODULE,
        str(snapshot_dir.resolve()),
        "--source-root",
        str(ROOT),
    ]
    run = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    if result.get("status") != "verified":
        raise ValueError("historical_seed627_verifier_failed")
    return {
        "result": result,
        "stdout_sha256": sha256(run.stdout.encode("utf-8")),
        "stderr_sha256": sha256(run.stderr.encode("utf-8")),
        "command": command,
    }


def _proposed_runtime_probe() -> dict[str, str]:
    code = (
        "import json,sys,numpy; "
        "print(json.dumps({'python':sys.version,'numpy':numpy.__version__},sort_keys=True))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(run.stdout)
    if (
        runtime.get("numpy") != "2.5.3"
        or runtime.get("numpy") != seed624.np.__version__
    ):
        raise ValueError("proposed_seed624_numpy_runtime_mismatch")
    runtime["probe_stdout_sha256"] = sha256(run.stdout.encode("utf-8"))
    return runtime


def authenticate_snapshot_bundle(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    """Authenticate exact pinned bytes and reconstruct Seed 627 evidence read-only."""
    for filename, expected in PINNED_FILES.items():
        raw = (snapshot_dir / filename).read_bytes()
        if sha256(raw) != expected:
            raise ValueError(f"pinned_seed627_file_hash_mismatch:{filename}")

    manifest = json.loads(
        (snapshot_dir / "source-identity-lineage-manifest.json").read_bytes()
    )
    historical_runtime = _historical_runtime_probe()
    _validate_historical_runtime(historical_runtime, manifest)
    historical_verification = _run_historical_seed627_verifier(snapshot_dir)

    proposed_runtime = _proposed_runtime_probe()
    evidence: dict[str, Any] = {}
    for name in NAMES:
        raw = (snapshot_dir / f"{name}.json").read_bytes()
        rows = json.loads(raw)
        # Cross-runtime verification on every row, not merely selected probes.
        for row in rows:
            cid, fid = seed624.identities(row["canonical_state"])
            if (
                cid != row["canonical_identity"]
                or fid != row["production_float32_identity"]
            ):
                raise ValueError(f"seed627_seed624_runtime_identity_mismatch:{name}")
        evidence[name] = {
            "authenticated": True,
            "rows": rows,
            "sha256": sha256(canonical_bytes(rows)),
            "file_sha256": sha256(raw),
            "manifest_sha256": PINNED_FILES["source-identity-lineage-manifest.json"],
            "lineage_sha256": sha256(canonical_bytes(manifest["training_lineage"])),
            "source_verification_sha256": sha256(
                canonical_bytes(manifest["source_verifier_results"])
            ),
        }
    prov = dict(manifest["provenance"])
    prov.update(
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
    prov["sha256"] = sha256(
        canonical_bytes(
            {key: value for key, value in prov.items() if key != "authenticated"}
        )
    )
    evidence["source_provenance"] = prov
    evidence["authentication"] = {
        "schema": "seed628-authenticated-seed627-adapter-v1",
        "historical_runtime_lock_sha256": HISTORICAL_RUNTIME_LOCK_SHA256,
        "historical_runtime": historical_runtime,
        "seed627_verification": historical_verification,
        "manifest_sha256": PINNED_FILES["source-identity-lineage-manifest.json"],
        "snapshot_file_sha256": {name: PINNED_FILES[f"{name}.json"] for name in NAMES},
        "identity_runtime": {
            "historical_numpy": historical_runtime["numpy"],
            "screening_numpy": proposed_runtime["numpy"],
            "screening_python": proposed_runtime["python"],
            "screening_probe_stdout_sha256": proposed_runtime["probe_stdout_sha256"],
        },
    }
    return evidence


def expected_snapshot_hashes(evidence: dict[str, Any]) -> dict[str, str]:
    return {name: evidence[name]["sha256"] for name in NAMES}


def expected_provenance_hash(evidence: dict[str, Any]) -> str:
    return evidence["source_provenance"]["sha256"]
