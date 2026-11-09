"""Verify seed560 evidence from an independent checkout with source hidden."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RELATIVE = Path("docs/data/seed560-paired-first-action/run-0004")


def _snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verify_physical_copy(materialized_tablebase: Path) -> dict:
    """Clone the committed tree, hide the execution worktree, and verify."""
    root = ROOT.resolve()
    materialized_tablebase = materialized_tablebase.resolve()
    source_asset_root = materialized_tablebase.parents[2]
    output = root / DATA_RELATIVE
    copy_root = root.parent / f"{root.name}-seed560-physical-copy-{os.getpid()}"
    hidden_root = root.with_name(f"{root.name}.hidden-for-seed560-verification")
    hidden_asset_root = source_asset_root.with_name(
        f"{source_asset_root.name}.hidden-for-seed560-verification"
    )
    if (
        copy_root.exists()
        or hidden_root.exists()
        or hidden_asset_root.exists()
        or not materialized_tablebase.is_file()
        or materialized_tablebase.is_relative_to(root)
        or source_asset_root == root
        or not (source_asset_root / "model-artifact/runtime/kalah_v1_21.kvtb").is_file()
    ):
        raise FileExistsError("seed560_physical_copy_path_already_exists")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "--branch",
            branch,
            str(root),
            str(copy_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    copied_output = copy_root / DATA_RELATIVE
    copied_tablebase = copy_root / ".tmp/seed560-assets/kalah_v1_21.kvtb"
    copied_tablebase.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(materialized_tablebase, copied_tablebase)
    source_before = _snapshot(output)
    copy_before = _snapshot(copied_output)
    child_code = (
        "import json; from pathlib import Path; "
        "from ml.alphazero_lite.verify_seed560_paired_first_action import verify; "
        f"blocked=Path({str(root)!r}); "
        f"blocked_asset_root=Path({str(source_asset_root)!r}); "
        "assert not blocked.exists(), 'original_execution_root_accessible'; "
        "assert not blocked_asset_root.exists(), 'original_asset_root_accessible'; "
        f"result=verify(Path({str(copied_output)!r}), "
        f"materialized_tablebase=Path({str(copied_tablebase)!r})); "
        "print(json.dumps(result, sort_keys=True))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(copy_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    cwd_before = Path.cwd()
    child: subprocess.CompletedProcess[str] | None = None
    try:
        os.chdir(root.parent)
        source_asset_root.rename(hidden_asset_root)
        root.rename(hidden_root)
        child = subprocess.run(
            [sys.executable, "-c", child_code],
            cwd=root.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if hidden_root.exists():
            hidden_root.rename(root)
        if hidden_asset_root.exists():
            hidden_asset_root.rename(source_asset_root)
        os.chdir(cwd_before)
    if child is None or child.returncode:
        raise RuntimeError(
            "seed560_physical_copy_verification_failed:"
            + (child.stderr if child else "child_not_started")
        )
    result = json.loads(child.stdout)
    if source_before != _snapshot(output) or copy_before != _snapshot(copied_output):
        raise ValueError("physical_copy_verification_mutated_evidence")
    receipt = {
        "schema": "seed560-physical-copy-verification-v1",
        "status": "passed",
        "copy_root": str(copy_root),
        "original_execution_root_hidden_during_verification": str(hidden_root),
        "original_execution_root_inaccessible": True,
        "original_materialized_tablebase_root": str(source_asset_root),
        "original_materialized_tablebase_root_inaccessible": True,
        "copied_materialized_tablebase": str(copied_tablebase),
        "child_verification": result,
        "source_evidence_files": len(source_before),
        "source_evidence_bytes_and_mtimes_unchanged": True,
        "copy_evidence_bytes_and_mtimes_unchanged": True,
        "child_stdout_sha256": hashlib.sha256(child.stdout.encode()).hexdigest(),
    }
    receipt_path = (
        root
        / "docs/data/seed560-paired-first-action/physical-copy-verification-review.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-tablebase", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_physical_copy(args.materialized_tablebase),
            sort_keys=True,
            indent=2,
        )
    )
