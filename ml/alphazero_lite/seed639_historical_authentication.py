"""Run Seed 615 historical input authentication in its pinned Python runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_PYTHON = Path("/home/alex/tmp/mancala-ai-venv/bin/python")
HISTORICAL_RUNTIME_LOCK = (
    ROOT
    / "docs/experiments/seed639-prospective-policy-prior-classification/historical-verifier-runtime-lock.txt"
)
HISTORICAL_RUNTIME_LOCK_SHA256 = (
    "99a376de59cfd861f9be09c34a16526e57f1c0177eb2a5186444e936a09347ec"
)
WORKTREE_SEED615_DEPENDENCY_LOCK = (
    ROOT / "ml/alphazero_lite/requirements-seed599-evaluation.txt"
)
DEFAULT_SEED615_ROOT = Path("/home/alex/Mancala/ai-seed615-authorized-eval")
SEED615_DEPENDENCY_LOCK_SHA256 = (
    "5950f891ea746be6058af5fae4c414ac0aca872bec064a7d28c383f795e2fa79"
)
EXPECTED_PYTHON_VERSION = "3.14.7"
EXPECTED_NUMPY_VERSION = "2.5.3"
EXPECTED_TORCH_VERSION = "2.14.0+cu130"
EXPECTED_THREAD_ENVIRONMENT = {
    "PYTHONHASHSEED": "615",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_packages() -> dict[str, str]:
    """Parse the exact environment inventory registered for the verifier."""
    if sha256_file(HISTORICAL_RUNTIME_LOCK) != HISTORICAL_RUNTIME_LOCK_SHA256:
        raise ValueError("seed639_historical_runtime_lock_hash_mismatch")
    packages = {}
    for line in HISTORICAL_RUNTIME_LOCK.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match is None:
            raise ValueError("seed639_historical_runtime_lock_line_invalid")
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        if normalized in packages:
            raise ValueError("seed639_historical_runtime_lock_duplicate_package")
        packages[normalized] = version
    if not packages:
        raise ValueError("seed639_historical_runtime_lock_empty")
    return packages


def runtime_probe(
    archive_path: Path, seed615_root: Path = DEFAULT_SEED615_ROOT
) -> dict[str, Any]:
    """Authenticate the separate historical interpreter without running analysis."""
    if Path(sys.executable) != HISTORICAL_PYTHON:
        raise RuntimeError("seed639_historical_python_path_mismatch")
    if (
        platform.python_version() != EXPECTED_PYTHON_VERSION
        or np.__version__ != EXPECTED_NUMPY_VERSION
        or str(torch.__version__) != EXPECTED_TORCH_VERSION
    ):
        raise RuntimeError("seed639_historical_runtime_version_mismatch")
    if torch.cuda.is_available():
        raise RuntimeError("seed639_historical_runtime_must_be_cpu_only")
    observed_environment = {
        key: os.environ.get(key) for key in EXPECTED_THREAD_ENVIRONMENT
    }
    if observed_environment != EXPECTED_THREAD_ENVIRONMENT:
        raise RuntimeError("seed639_historical_thread_environment_mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("seed639_historical_torch_determinism_mismatch")

    expected_packages = _expected_packages()
    installed_packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    if installed_packages != expected_packages:
        raise RuntimeError("seed639_historical_runtime_package_inventory_mismatch")

    seed615_lock = (
        seed615_root / "ml/alphazero_lite/requirements-seed599-evaluation.txt"
    )
    if (
        sha256_file(WORKTREE_SEED615_DEPENDENCY_LOCK) != SEED615_DEPENDENCY_LOCK_SHA256
        or sha256_file(seed615_lock) != SEED615_DEPENDENCY_LOCK_SHA256
    ):
        raise ValueError("seed639_seed615_dependency_lock_hash_mismatch")
    with tarfile.open(archive_path, mode="r:*") as archive:
        archive_members = [member.name for member in archive.getmembers()]
    if "population/population.json" not in archive_members:
        raise ValueError(
            "seed639_seed636_archive_member_unavailable_in_verifier_runtime"
        )

    return {
        "python_executable": str(Path(sys.executable)),
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "cuda_available": False,
        "device": "cpu",
        "thread_environment": observed_environment,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "package_inventory": installed_packages,
        "runtime_lock_path": str(HISTORICAL_RUNTIME_LOCK),
        "runtime_lock_sha256": sha256_file(HISTORICAL_RUNTIME_LOCK),
        "seed615_dependency_lock_path": str(seed615_lock),
        "seed615_dependency_lock_sha256": SEED615_DEPENDENCY_LOCK_SHA256,
        "worktree_seed615_dependency_lock_path": str(WORKTREE_SEED615_DEPENDENCY_LOCK),
        "worktree_seed615_dependency_lock_sha256": SEED615_DEPENDENCY_LOCK_SHA256,
        "archive_read_path": str(archive_path),
        "archive_member_count": len(archive_members),
        "archive_population_member_present": True,
    }


def authenticate_seed615(seed615_root: Path, archive_path: Path) -> dict[str, Any]:
    """Run the complete historical verifier after the runtime probe succeeds."""
    runtime = runtime_probe(archive_path, seed615_root)
    from ml.alphazero_lite.seed639_seed615_historical_adapter import authenticate

    authentication = authenticate(seed615_root.resolve())
    return {
        "schema": "seed639-seed615-historical-authentication-v1",
        "runtime": runtime,
        "authentication": authentication,
    }


def main() -> None:
    """Expose a read-only runtime probe and historical authentication command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--seed615-root", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--evidence-output",
        type=Path,
        help="optionally preserve a successful read-only authentication receipt create-only",
    )
    args = parser.parse_args()
    seed615_root = args.seed615_root or DEFAULT_SEED615_ROOT
    if args.probe_only:
        result = {
            "runtime": runtime_probe(args.archive, seed615_root),
            "authentication": None,
        }
    else:
        if args.seed615_root is None:
            parser.error("--seed615-root is required unless --probe-only is used")
        result = authenticate_seed615(seed615_root, args.archive)
    payload = (
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if args.evidence_output is not None:
        with args.evidence_output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        digest_path = args.evidence_output.with_suffix(
            args.evidence_output.suffix + ".sha256"
        )
        with digest_path.open("xb") as stream:
            stream.write(
                f"{hashlib.sha256(payload).hexdigest()}  {args.evidence_output.name}\n".encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
