"""Copy and verify seed556 evidence with source and tablebase paths hidden."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verify_physical_copy(
    *,
    root: Path,
    output: Path,
    copy_root: Path,
    materialized_tablebase: Path,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    root, output, copy_root = root.resolve(), output.resolve(), copy_root.resolve()
    materialized_tablebase = materialized_tablebase.resolve()
    hidden_root = root.with_name(
        root.name + ".hidden-for-seed556-physical-verification"
    )
    hidden_tablebase = materialized_tablebase.with_name(
        materialized_tablebase.name + ".hidden-for-seed556-physical-verification"
    )
    if copy_root.exists() or hidden_root.exists() or hidden_tablebase.exists():
        raise FileExistsError("seed556_physical_copy_destination_or_hidden_path_exists")
    copy_root.mkdir(parents=True)
    shutil.copytree(root / "ml/alphazero_lite", copy_root / "ml/alphazero_lite")
    shutil.copytree(
        root / "docs/data/seed511-fixed-composition-screen-v4",
        copy_root / "docs/data/seed511-fixed-composition-screen-v4",
    )
    copied_output = copy_root / output.relative_to(root)
    shutil.copytree(output, copied_output)
    shutil.copytree(
        root / "model-artifact/runtime", copy_root / "model-artifact/runtime"
    )
    copied_tablebase = copy_root / "physical-assets/kalah_v1_21.kvtb"
    copied_tablebase.parent.mkdir(parents=True)
    shutil.copy2(materialized_tablebase, copied_tablebase)

    source_before = _snapshot(output)
    copy_before = _snapshot(copied_output)
    correction = (
        copy_root / "ml/alphazero_lite/verify_seed556_completion_order_correction.py"
    )
    child = (
        "import json; from pathlib import Path; "
        "from ml.alphazero_lite.verify_seed556_completion_order_correction import verify; "
        f"blocked=[Path({str(root)!r}),Path({str(materialized_tablebase)!r})]; "
        "assert all(not p.exists() for p in blocked), 'original_paths_accessible'; "
        f"result=verify(Path({str(copy_root)!r}),Path({str(copied_output)!r}), "
        f"materialized_tablebase=Path({str(copied_tablebase)!r}), "
        f"native_probe=Path({str(copy_root / 'model-artifact/runtime/kalah_v1_tablebase')!r})); "
        "print(json.dumps(result,sort_keys=True))"
    )
    env = {**os.environ, "PYTHONPATH": str(copy_root)}
    cwd_before = Path.cwd()
    child_result: subprocess.CompletedProcess[str] | None = None
    try:
        os.chdir(root.parent)
        root.rename(hidden_root)
        materialized_tablebase.rename(hidden_tablebase)
        child_result = subprocess.run(
            [sys.executable, "-c", child],
            cwd=root.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if hidden_tablebase.exists():
            hidden_tablebase.rename(materialized_tablebase)
        if hidden_root.exists():
            hidden_root.rename(root)
        os.chdir(cwd_before)
    if child_result is None or child_result.returncode != 0:
        raise RuntimeError(
            "seed556_physical_copy_verification_failed:"
            + (child_result.stderr if child_result else "child_not_started")
        )
    result = json.loads(child_result.stdout)
    source_after = _snapshot(output)
    copy_after = _snapshot(copied_output)
    if source_before != source_after or copy_before != copy_after:
        raise ValueError("seed556_verification_changed_bytes_or_mtimes")
    receipt = {
        "schema": "seed556_physical_copy_verification_v1",
        "status": "passed",
        "copy_root": str(copy_root),
        "copied_materialized_tablebase": str(copied_tablebase),
        "original_root_hidden": str(hidden_root),
        "original_tablebase_hidden": str(hidden_tablebase),
        "original_paths_inaccessible_in_child": True,
        "child_exit_status": child_result.returncode,
        "child_verification": result,
        "source_evidence_files": len(source_before),
        "source_bytes_and_mtimes_unchanged": True,
        "physical_copy_bytes_and_mtimes_unchanged": True,
        "correction_verifier_sha256": hashlib.sha256(
            correction.read_bytes()
        ).hexdigest(),
        "stdout_sha256": hashlib.sha256(child_result.stdout.encode()).hexdigest(),
    }
    receipt_path = receipt_path or output / "physical-copy-verification.json"
    temporary = receipt_path.with_name(f".{receipt_path.name}.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)
    return receipt
