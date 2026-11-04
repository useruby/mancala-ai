"""Verify seed553 evidence in a physical copy with the source worktree hidden."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed553-fixed-state-budget")
CORPUS = Path("docs/data/seed549-matched-selfplay")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def copy_inventory(root: Path) -> list[str]:
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    receipt = json.loads((root / CORPUS / "post-generation-receipt.json").read_text())
    paths = set(registration["source_sha256"])
    paths.update(
        {(OUT / path.name).as_posix() for path in out.iterdir() if path.is_file()}
    )
    paths.update(
        {
            (CORPUS / "registration.json").as_posix(),
            (CORPUS / "post-generation-receipt.json").as_posix(),
            (CORPUS / "control-source-manifest.json").as_posix(),
            (CORPUS / "control-training-row-identities.jsonl").as_posix(),
            Path("docs/data/seed553-prerequisite-failure.json").as_posix(),
        }
    )
    paths.update((CORPUS / path).as_posix() for path in receipt["products"])
    paths.update(
        (CORPUS / "control" / f"shard-{shard:03d}.jsonl").as_posix()
        for shard in range(6)
    )
    paths.add("docs/data/seed542-selfplay-4096/initializer.npz")
    return sorted(paths)


def snapshot(root: Path, inventory: list[str]) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": sha(root / relative),
            "mtime_ns": (root / relative).stat().st_mtime_ns,
            "bytes": (root / relative).stat().st_size,
        }
        for relative in inventory
    }


def run_copy_verification(
    source_root: Path,
    copy_root: Path,
    hidden_root: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    copy_root = copy_root.resolve()
    hidden_root = hidden_root.resolve()
    if copy_root == source_root or source_root in copy_root.parents:
        raise ValueError("seed553_copy_must_be_outside_source_worktree")
    if copy_root.exists() or hidden_root.exists():
        raise FileExistsError("seed553_physical_copy_target_already_exists")
    inventory = copy_inventory(source_root)
    source_snapshot = snapshot(source_root, inventory)
    copy_root.mkdir(parents=True)
    for relative in inventory:
        source = source_root / relative
        destination = copy_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    copied_snapshot = snapshot(copy_root, inventory)
    if copied_snapshot != source_snapshot:
        raise ValueError("seed553_physical_copy_bytes_or_mtime_mismatch")

    registration = json.loads((copy_root / OUT / "registration.json").read_text())
    interpreter = registration["runtime"]["interpreter"]
    command = [
        *registration["resource_launcher"],
        interpreter,
        "-m",
        "ml.alphazero_lite.verify_seed553_fixed_state_budget",
        "--root",
        str(copy_root),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(copy_root)
    source_root.rename(hidden_root)
    result: subprocess.CompletedProcess[str] | None = None
    restore_error = None
    try:
        if source_root.exists():
            raise RuntimeError("seed553_original_worktree_path_still_accessible")
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
    if restore_error is not None:
        raise RuntimeError(f"seed553_original_worktree_restore_failed:{restore_error}")
    if result is None:
        raise RuntimeError("seed553_physical_verifier_not_run")
    copied_after = snapshot(copy_root, inventory)
    if copied_after != source_snapshot:
        raise RuntimeError("seed553_physical_verifier_modified_evidence")
    original_after = snapshot(source_root, inventory)
    if original_after != source_snapshot:
        raise RuntimeError("seed553_source_evidence_changed_during_copy_verification")
    receipt = {
        "schema": "seed553-physical-copy-verification-v1",
        "status": "passed" if result.returncode == 0 else "failed",
        "source_worktree_hidden_during_verifier": True,
        "source_worktree_path_absent_during_verifier": True,
        "copy_root": str(copy_root),
        "verifier_cwd": str(copy_root.parent),
        "command": command,
        "exit_status": result.returncode,
        "verifier_stdout": result.stdout,
        "verifier_stderr": result.stderr,
        "evidence_files": source_snapshot,
        "copy_bytes_and_mtimes_preserved": copied_after == source_snapshot,
        "source_bytes_and_mtimes_preserved": original_after == source_snapshot,
    }
    atomic_json(source_root / OUT / "physical-copy-verification.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--copy-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_copy_verification(args.source_root, args.copy_root, args.hidden_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["exit_status"] != 0:
        raise SystemExit(result["exit_status"])


if __name__ == "__main__":
    main()
