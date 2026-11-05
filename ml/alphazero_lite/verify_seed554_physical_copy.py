"""Run seed554's read-only verifier from a physical copy with the source hidden."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed554-fixed-state-teacher-budget")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[str]:
    freeze = json.loads((root / OUT / "label-generation-freeze.json").read_text())
    paths = set(freeze["sources"])
    paths.add("ml/alphazero_lite/verify_seed549_training.py")
    paths.update(
        path.relative_to(root).as_posix() for path in (root / "ml").rglob("*.py")
    )
    paths.add("docs/data/seed542-selfplay-4096/initializer.npz")
    paths.add("docs/data/seed553-fixed-state-budget/registration.json")
    paths.update(
        path.relative_to(root).as_posix()
        for path in (root / "docs/data/seed549-matched-selfplay").rglob("*")
        if path.is_file()
    )
    paths.update(
        path.relative_to(root).as_posix()
        for path in (root / OUT).rglob("*")
        if path.is_file()
    )
    return sorted(paths)


def snapshot(root: Path, files: list[str]) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": sha(root / relative),
            "mtime_ns": (root / relative).stat().st_mtime_ns,
            "bytes": (root / relative).stat().st_size,
        }
        for relative in files
    }


def run_copy(source_root: Path, copy_root: Path, hidden_root: Path) -> dict[str, Any]:
    source_root, copy_root, hidden_root = (
        source_root.resolve(),
        copy_root.resolve(),
        hidden_root.resolve(),
    )
    if copy_root == source_root or source_root in copy_root.parents:
        raise ValueError("seed554_copy_must_be_outside_source_worktree")
    if copy_root.exists() or hidden_root.exists():
        raise FileExistsError("seed554_physical_copy_target_already_exists")

    files = inventory(source_root)
    original_before = snapshot(source_root, files)
    copy_root.mkdir(parents=True)
    for relative in files:
        source = source_root / relative
        destination = copy_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    copy_before = snapshot(copy_root, files)
    if copy_before != original_before:
        raise ValueError("seed554_physical_copy_bytes_or_mtimes_mismatch")

    command = [
        "taskset",
        "-c",
        "31",
        "nice",
        "-n",
        "19",
        sys.executable,
        "-m",
        "ml.alphazero_lite.verify_seed554_fixed_state_teacher_budget",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(copy_root)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"

    source_root.rename(hidden_root)
    result: subprocess.CompletedProcess[str] | None = None
    restore_error: str | None = None
    try:
        if source_root.exists():
            raise RuntimeError("seed554_original_worktree_path_still_accessible")
        result = subprocess.run(
            command,
            cwd=copy_root.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            hidden_root.rename(source_root)
        except OSError as error:
            restore_error = repr(error)
    if restore_error:
        raise RuntimeError(f"seed554_original_worktree_restore_failed:{restore_error}")
    if result is None:
        raise RuntimeError("seed554_physical_copy_verifier_not_run")
    copy_after = snapshot(copy_root, files)
    original_after = snapshot(source_root, files)
    if copy_after != copy_before:
        raise RuntimeError("seed554_physical_verifier_modified_copy_evidence")
    if original_after != original_before:
        raise RuntimeError("seed554_physical_verification_changed_original_evidence")

    receipt = {
        "schema": "seed554-physical-copy-verification-v1",
        "status": "passed" if result.returncode == 0 else "failed",
        "source_root_hidden_during_verifier": True,
        "source_root_path_absent_during_verifier": True,
        "copy_root": str(copy_root),
        "verifier_cwd": str(copy_root.parent),
        "command": command,
        "exit_status": result.returncode,
        "verifier_stdout": result.stdout,
        "verifier_stderr": result.stderr,
        "evidence_snapshot": original_before,
        "copy_bytes_and_mtimes_unchanged": copy_after == copy_before,
        "source_bytes_and_mtimes_unchanged": original_after == original_before,
    }
    destination = source_root / OUT / "physical-copy-verification.json"
    temporary = destination.with_suffix(".json.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--copy-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_copy(args.source_root, args.copy_root, args.hidden_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["exit_status"]:
        raise SystemExit(receipt["exit_status"])


if __name__ == "__main__":
    main()
