"""Independent read-only verifier for Seed 604 runtime certification v1."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed604_training_executor import RUNTIME

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("docs/data/seed604-replay-optimization/runtime-certification-v3")
SOURCE_PATHS = (
    "ml/alphazero_lite/seed604_training_executor.py",
    "ml/alphazero_lite/seed604_replay_optimization.py",
    "ml/alphazero_lite/seed604_bounded_exclusions.py",
    "ml/alphazero_lite/verify_seed604_opening_cohort.py",
    "ml/alphazero_lite/test_seed604_training_executor.py",
    "ml/alphazero_lite/publish_seed604_runtime_certification.py",
    "ml/alphazero_lite/verify_seed604_runtime_certification.py",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(name: str) -> bytes:
    path = ROOT / ARTIFACTS / name
    if not path.is_file():
        raise ValueError(f"missing_certification_file:{name}")
    return path.read_bytes()


def verify() -> dict[str, Any]:
    manifest = json.loads(_read("sha256-manifest.json"))
    if manifest.get("schema") != "seed604-runtime-certification-digests-v1":
        raise ValueError("digest_manifest_schema_mismatch")
    files = manifest.get("files")
    expected_files = {
        "README.md",
        "environment.json",
        "requirements-lock.txt",
        "runtime-identity.json",
        "synthetic-test-results.json",
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("digest_manifest_file_set_mismatch")
    for name, digest in files.items():
        if sha256(_read(name)) != digest:
            raise ValueError(f"certification_digest_mismatch:{name}")

    runtime = json.loads(_read("runtime-identity.json"))
    if runtime.get("status") != "runtime-identity-verified":
        raise ValueError("runtime_identity_not_verified")
    if runtime.get("runtime_contract") != RUNTIME:
        raise ValueError("registered_runtime_contract_mismatch")
    observed = runtime.get("observed_runtime", {})
    observed_identity = {
        "python": observed.get("python"),
        "numpy": next(
            (
                item.get("version")
                for item in runtime.get("package_wheels", [])
                if item.get("name", "").lower() == "numpy"
            ),
            None,
        ),
        "torch": next(
            (
                item.get("version")
                for item in runtime.get("package_wheels", [])
                if item.get("name", "").lower() == "torch"
            ),
            None,
        ),
        "device": observed.get("cpu_model_execution_device"),
        "threads": observed.get("torch_num_threads"),
        "pythonhashseed": observed.get("environment", {}).get("PYTHONHASHSEED"),
        "omp_num_threads": observed.get("environment", {}).get("OMP_NUM_THREADS"),
        "mkl_num_threads": observed.get("environment", {}).get("MKL_NUM_THREADS"),
        "deterministic_algorithms": observed.get("deterministic_algorithms"),
    }
    if observed_identity != RUNTIME:
        raise ValueError("observed_runtime_does_not_match_contract")
    if (
        observed.get("implementation") != "CPython"
        or observed.get("machine") != "x86_64"
    ):
        raise ValueError("python_or_platform_identity_mismatch")
    source_hashes = runtime.get("source_sha256", {})
    if set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("source_binding_set_mismatch")
    for relative in SOURCE_PATHS:
        if sha256((ROOT / relative).read_bytes()) != source_hashes[relative]:
            raise ValueError(f"source_binding_mismatch:{relative}")
    commit = runtime.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("source_commit_binding_missing")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    lock = _read("requirements-lock.txt").decode()
    for package in runtime.get("package_wheels", []):
        requirement = (
            f"{package['name']}=={package['version']} "
            f"--hash=sha256:{package['wheel_sha256']}"
        )
        if requirement not in lock.splitlines():
            raise ValueError(f"dependency_lock_binding_mismatch:{package['name']}")

    tests = json.loads(_read("synthetic-test-results.json"))
    if (
        tests.get("tests_run") != 46
        or tests.get("failures")
        or tests.get("errors")
        or tests.get("skipped")
        or tests.get("expected_executor_tests")
        != {"dependency_light": 12, "pytorch_synthetic": 6}
        or tests.get("production_dataset_training") is not False
    ):
        raise ValueError("synthetic_gate_evidence_mismatch")
    if runtime.get("historical_b512_bitwise_reproduction") != (
        "not performed; remains unverified"
    ):
        raise ValueError("historical_reproduction_scope_mismatch")
    return {
        "passed": True,
        "runtime_identity_verified": True,
        "synthetic_gates_verified": True,
        "b512_bitwise_reproduction": "not performed; remains unverified",
        "files_verified": len(files),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
