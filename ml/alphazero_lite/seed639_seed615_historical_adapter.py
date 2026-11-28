"""Run the original Seed 615 verifier from its authenticated checkout."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

FREEZE_SHA256 = "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
SERVING = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
FROZEN_RESULTS = {
    "B512_score": 0.502197265625,
    "S615_score": 0.496337890625,
    "delta": -0.005859375,
    "interval": [-0.01611328125, 0.00439453125],
    "S615_vs_P_E4_lower_bound": 0.486572265625,
    "interpretation": "neither",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_imports(root: Path) -> tuple[Any, Any, dict[str, str]]:
    """Import verifier modules only after making the historical checkout primary."""
    root = root.resolve(strict=True)
    seed639_root = Path(__file__).resolve().parents[2]

    def resolved_entry(entry: str | None) -> Path:
        return Path(entry or ".").resolve()

    sys.path[:] = [
        str(root),
        *[
            entry
            for entry in sys.path
            if resolved_entry(entry) != root and resolved_entry(entry) != seed639_root
        ],
    ]
    for name in tuple(sys.modules):
        if name == "ml" or name.startswith("ml."):
            del sys.modules[name]
    importlib.invalidate_caches()

    contract = importlib.import_module("ml.alphazero_lite.seed615_evaluation")
    verifier = importlib.import_module("ml.alphazero_lite.verify_seed615_evaluation")
    origins: dict[str, str] = {}
    for module in (contract, verifier):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError("seed639_historical_module_origin_missing")
        origin = Path(module_file).resolve(strict=True)
        try:
            origin.relative_to(root)
        except ValueError as error:
            raise RuntimeError("seed639_historical_module_origin_mismatch") from error
        origins[module.__name__] = str(origin)
    return contract, verifier, origins


def _historical_module_identities(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Bind every imported project module to a file within the historical root."""
    origins: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        if name != "ml.alphazero_lite" and not name.startswith("ml.alphazero_lite."):
            continue
        raw_origin = getattr(module, "__file__", None)
        if raw_origin is None:
            continue
        origin = Path(raw_origin).resolve(strict=True)
        try:
            relative = origin.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"seed639_historical_module_origin_mismatch:{name}"
            ) from error
        origins[name] = str(origin)
        source_hashes[relative] = _sha256(origin)
    if not origins:
        raise RuntimeError("seed639_historical_modules_not_observed")
    return origins, source_hashes


def authenticate(root: Path) -> dict[str, Any]:
    """Reproduce the existing read-only Seed 615 authentication checks."""
    root = root.resolve(strict=True)
    contract, verifier, origins = _historical_imports(root)
    evidence = (
        root / "docs/data/seed615-source-opening-diversity/evaluation-preparation"
    )
    freeze = evidence / "execution-freeze-v4.json"
    if _sha256(freeze) != FREEZE_SHA256:
        raise ValueError("evaluation_freeze_identity_mismatch")
    verification = verifier.verify_results(root)
    origins, source_hashes = _historical_module_identities(root)
    expected_verification = {
        "passed": True,
        "completed_games": 4096,
        "capped_games": 0,
        "evaluation_authorized": True,
        "analysis_present": True,
    }
    if verification != expected_verification:
        raise ValueError("independent_evaluation_verification_mismatch")

    freeze_doc = json.loads(freeze.read_bytes())
    checkpoint_paths = {
        name: root / freeze_doc["registration"]["models"][name]["path"]
        for name in CHECKPOINTS
    }
    for name, expected in CHECKPOINTS.items():
        if _sha256(checkpoint_paths[name]) != expected:
            raise ValueError(f"checkpoint_identity_mismatch:{name}")
        runtime_path = (
            root
            / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
            / name
            / "model.npz"
        )
        if _sha256(runtime_path) != expected:
            raise ValueError(f"runtime_checkpoint_identity_mismatch:{name}")

    if freeze_doc.get("serving_model_sha256") != SERVING:
        raise ValueError("serving_checkpoint_identity_mismatch")
    analysis_path = evidence / "analysis.json"
    analysis = json.loads(analysis_path.read_bytes())
    actual = analysis["results"]
    observed = {
        "B512_score": actual["B512_score"],
        "S615_score": actual["S615_score"],
        "delta": actual["S615_score"] - actual["B512_score"],
        "interval": actual["S615_minus_B512_interval"],
        "S615_vs_P_E4_lower_bound": actual["S615_vs_P_E4_one_sided_lower_bound"],
        "interpretation": actual["interpretation"],
    }
    if observed != FROZEN_RESULTS:
        raise ValueError("published_analysis_identity_or_result_mismatch")

    for relative, field in (
        (contract.OPENINGS, "file_sha256"),
        (contract.ASSIGNMENTS, "file_sha256_assignments"),
    ):
        if _sha256(root / relative) != freeze_doc["cohort"][field]:
            raise ValueError(f"registered_manifest_identity_mismatch:{relative}")

    return {
        "verification": verification,
        "freeze_sha256": FREEZE_SHA256,
        "checkpoint_sha256": {**CHECKPOINTS, "P-E4": SERVING},
        "analysis_sha256": _sha256(analysis_path),
        "opening_manifest_sha256": freeze_doc["cohort"]["file_sha256"],
        "assignment_manifest_sha256": freeze_doc["cohort"]["file_sha256_assignments"],
        "historical_module_origins": origins,
        "historical_source_sha256": source_hashes,
    }
