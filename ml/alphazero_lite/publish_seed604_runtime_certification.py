"""Create a versioned, create-only Seed 604 runtime certification packet."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed604_training_executor import RUNTIME, check_runtime

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path("docs/data/seed604-replay-optimization/runtime-certification-v2")
WHEELS = Path(".tmp/seed604-runtime/wheels")
SOURCE_PATHS = (
    "ml/alphazero_lite/seed604_training_executor.py",
    "ml/alphazero_lite/seed604_replay_optimization.py",
    "ml/alphazero_lite/seed604_bounded_exclusions.py",
    "ml/alphazero_lite/verify_seed604_opening_cohort.py",
    "ml/alphazero_lite/test_seed604_training_executor.py",
    "ml/alphazero_lite/publish_seed604_runtime_certification.py",
    "ml/alphazero_lite/verify_seed604_runtime_certification.py",
)
TEST_MODULES = (
    "ml.alphazero_lite.test_seed604_training_executor",
    "ml.alphazero_lite.test_seed604_bounded_exclusions",
    "ml.alphazero_lite.test_seed604_replay_optimization",
    "ml.alphazero_lite.test_seed604_opening_cohort",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)


def _run_tests() -> dict[str, Any]:
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for module in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module))
    result = unittest.TestResult()
    suite.run(result)
    expected = ("Ran 46 tests",)
    report = {
        "modules": list(TEST_MODULES),
        "tests_run": result.testsRun,
        "failures": [str(test) + ": " + detail for test, detail in result.failures],
        "errors": [str(test) + ": " + detail for test, detail in result.errors],
        "skipped": [[str(test), reason] for test, reason in result.skipped],
        "expected_executor_tests": {"dependency_light": 12, "pytorch_synthetic": 6},
        "production_dataset_training": False,
        "execution_freeze_created": False,
        "evaluation_games_started": False,
    }
    if result.testsRun != 46 or result.failures or result.errors or result.skipped:
        raise RuntimeError(f"synthetic_test_gate_failed:{report!r}")
    if expected[0] != f"Ran {result.testsRun} tests":
        raise RuntimeError("synthetic_test_count_unexpected")
    return report


def _wheel_inventory() -> tuple[list[dict[str, Any]], str]:
    wheel_paths = sorted(WHEELS.glob("*.whl"))
    by_normalized_name: dict[str, Path] = {}
    for path in wheel_paths:
        parts = path.name.split("-")
        if len(parts) < 5:
            raise RuntimeError(f"invalid_wheel_filename:{path.name}")
        by_normalized_name[parts[0].replace("_", "-").lower()] = path
    entries: list[dict[str, Any]] = []
    lock_lines: list[str] = []
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        name = dist.metadata["Name"]
        version = dist.version
        normalized = name.replace("_", "-").lower()
        wheel = by_normalized_name.get(normalized)
        if wheel is None:
            raise RuntimeError(
                f"wheel_missing_for_installed_distribution:{name}=={version}"
            )
        digest = _sha256(wheel.read_bytes())
        entries.append(
            {
                "name": name,
                "version": version,
                "wheel_filename": wheel.name,
                "wheel_sha256": digest,
                "distribution_origin": str(dist.locate_file("")),
            }
        )
        lock_lines.append(f"{name}=={version} --hash=sha256:{digest}")
    lock = (
        "# SHA-256 locked installed environment; CPython 3.14.7, Linux x86_64.\n"
        + "\n".join(lock_lines)
        + "\n"
    )
    return entries, lock


def publish() -> Path:
    import numpy
    import torch

    if (ROOT / DESTINATION).exists():
        raise FileExistsError(f"create_only_destination_exists:{DESTINATION}")
    if sys.version_info[:3] != (3, 14, 7) or numpy.__version__ != "2.5.2":
        raise RuntimeError("python_or_numpy_runtime_mismatch")
    check_runtime(dict(RUNTIME), torch)
    if not os.environ.get("PYTHONHASHSEED") == "599":
        raise RuntimeError("pythonhashseed_mismatch")

    destination = ROOT / DESTINATION
    destination.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    source_bindings = {
        relative: _sha256((ROOT / relative).read_bytes()) for relative in SOURCE_PATHS
    }
    wheels, lock = _wheel_inventory()
    test_results = _run_tests()
    identity = {
        "status": "runtime-identity-verified",
        "runtime_contract": dict(RUNTIME),
        "observed_runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "executable_sha256": _sha256(Path(sys.executable).resolve().read_bytes()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model_execution_device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "torch_build_configuration": torch.__config__.show(),
        },
        "git_commit": commit,
        "source_sha256": source_bindings,
        "package_wheels": wheels,
        "cpu_numerical_backend": "MKL, oneDNN, OpenMP; AVX2 reported by PyTorch",
        "historical_b512_bitwise_reproduction": "not performed; remains unverified",
    }
    files: dict[str, bytes] = {
        "requirements-lock.txt": lock.encode(),
        "runtime-identity.json": json.dumps(identity, indent=2, sort_keys=True).encode()
        + b"\n",
        "synthetic-test-results.json": json.dumps(
            test_results, indent=2, sort_keys=True
        ).encode()
        + b"\n",
        "environment.json": json.dumps(
            {
                "python": "CPython==3.14.7",
                "numpy": "2.5.2",
                "torch": "2.13.0+cu130",
                "torch_wheel_index": "https://download.pytorch.org/whl/cu130/torch/",
                "cpu_only_execution": True,
                "environment_variables": {
                    "PYTHONHASHSEED": "599",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                "reproduction": [
                    "python3.14 -m venv .venv",
                    "export PYTHONHASHSEED=599 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
                    "python -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-lock.txt",
                    "python -m ml.alphazero_lite.verify_seed604_runtime_certification",
                ],
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
    }
    files["README.md"] = (
        "# Seed 604 runtime certification v2\n\n"
        "Runtime identity verified: required software versions and deterministic settings "
        "authenticated in a fresh process. Synthetic gates verified: all six PyTorch "
        "synthetic tests executed successfully; 12 dependency-light tests also passed. "
        "B512 bitwise reproduction was not performed and remains unverified. No production "
        "dataset training, B2048 checkpoint generation, freeze, authorization, or evaluation "
        "games occurred.\n\n"
        "Reproduce from the recorded Python 3.14.7 CPython installation using the hashed "
        "requirements lock and environment specification. This certifies runtime identity "
        "and synthetic guards only; matching versions do not establish historical numerical "
        "reproduction.\n"
    ).encode()
    for name, raw in files.items():
        _write(destination / name, raw)
    digest_manifest = {
        "schema": "seed604-runtime-certification-digests-v1",
        "files": {name: _sha256(raw) for name, raw in sorted(files.items())},
    }
    _write(
        destination / "sha256-manifest.json",
        json.dumps(digest_manifest, indent=2, sort_keys=True).encode() + b"\n",
    )
    return destination


if __name__ == "__main__":
    print(publish())
