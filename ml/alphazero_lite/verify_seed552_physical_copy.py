"""Verify seed552 from a byte-for-byte copy while the source worktree is absent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

OUT = Path("docs/data/seed552-c108-serving-p")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_copy(
    root: Path,
    copy_root: Path,
    hidden_root: Path,
    python: Path,
    *,
    tablebase_path: Path | None = None,
    frozen_execution_root: Path | None = None,
    frozen_hidden_root: Path | None = None,
) -> dict[str, Any]:
    root, copy_root, hidden_root = (
        root.resolve(),
        copy_root.resolve(),
        hidden_root.resolve(),
    )
    if copy_root.exists() or hidden_root.exists():
        raise FileExistsError("seed552_physical_copy_destination_must_be_absent")
    original_output = root / OUT
    before = inventory(original_output)
    shutil.copytree(
        root,
        copy_root,
        ignore=shutil.ignore_patterns(
            ".git", ".tmp", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc"
        ),
        copy_function=shutil.copy2,
    )
    copied_output = copy_root / OUT
    copied_before = inventory(copied_output)
    if copied_before != before:
        raise ValueError("seed552_physical_copy_bytes_or_mtimes_mismatch_before_verify")
    frozen_execution_root = (
        frozen_execution_root.resolve() if frozen_execution_root else None
    )
    frozen_hidden_root = frozen_hidden_root.resolve() if frozen_hidden_root else None
    frozen_before = (
        inventory(frozen_execution_root / OUT) if frozen_execution_root else None
    )
    if frozen_execution_root and (
        not frozen_execution_root.is_dir()
        or frozen_hidden_root is None
        or frozen_hidden_root.exists()
    ):
        raise ValueError("seed552_frozen_execution_root_not_ready_for_isolation")
    source_tablebase = (
        tablebase_path or root / "model-artifact/runtime/kalah_v1_21.kvtb"
    )
    copied_runtime_dir = copy_root / ".tmp/seed552-runtime"
    copied_runtime_dir.mkdir(parents=True)
    copied_tablebase = copied_runtime_dir / "kalah_v1_21.kvtb"
    shutil.copy2(source_tablebase, copied_tablebase)

    os.chdir(Path.home())
    moved_frozen = False
    moved_root = False
    try:
        if frozen_execution_root:
            os.rename(frozen_execution_root, frozen_hidden_root)
            moved_frozen = True
        os.rename(root, hidden_root)
        moved_root = True
        if root.exists() or (frozen_execution_root and frozen_execution_root.exists()):
            raise ValueError("seed552_original_worktree_still_accessible")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(copy_root)
        command = [
            str(python),
            "-m",
            "ml.alphazero_lite.verify_seed552_c108_serving_p",
            "--root",
            str(copy_root),
            "--out",
            str(copied_output),
            "--tablebase",
            str(copied_tablebase),
            "--allow-relocated-copy",
        ]
        result = subprocess.run(
            command,
            cwd=Path.home(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    finally:
        if moved_root:
            os.rename(hidden_root, root)
        if moved_frozen:
            os.rename(frozen_hidden_root, frozen_execution_root)

    if not root.exists() or hidden_root.exists():
        raise ValueError("seed552_original_worktree_restore_failed")
    original_after = inventory(original_output)
    copied_after = inventory(copied_output)
    frozen_after = (
        inventory(frozen_execution_root / OUT) if frozen_execution_root else None
    )
    if original_after != before:
        raise ValueError("seed552_original_evidence_bytes_or_mtimes_changed")
    if copied_after != copied_before:
        raise ValueError("seed552_physical_copy_evidence_bytes_or_mtimes_changed")
    if frozen_before is not None and frozen_after != frozen_before:
        raise ValueError("seed552_frozen_execution_evidence_bytes_or_mtimes_changed")
    verified = json.loads(result.stdout)
    if (
        verified.get("status") != "verified_complete"
        or verified.get("validated_games") != 2048
        or verified.get("validated_pairs") != 1024
    ):
        raise ValueError("seed552_physical_copy_verification_incomplete")
    receipt = {
        "schema": "seed552_physical_copy_verification_v1",
        "status": "verified_complete",
        "source_worktree": str(root),
        "physical_copy": str(copy_root),
        "verifier_cwd": str(Path.home()),
        "source_worktree_inaccessible_during_verification": True,
        "frozen_execution_root": str(frozen_execution_root)
        if frozen_execution_root
        else str(root),
        "frozen_execution_root_inaccessible_during_verification": True,
        "tablebase_override_sha256": file_sha256(copied_tablebase),
        "verified_games": verified["validated_games"],
        "verified_pairs": verified["validated_pairs"],
        "decision": verified["analysis"]["decision"],
        "score": verified["analysis"]["score"],
        "one_sided_95_lower": verified["analysis"]["one_sided_95_lower"],
        "descriptive_95_interval": verified["analysis"]["descriptive_95_interval"],
        "terminal_wins": verified["analysis"]["terminal_wins"],
        "terminal_draws": verified["analysis"]["terminal_draws"],
        "terminal_losses": verified["analysis"]["terminal_losses"],
        "capped_nonterminal_games": verified["analysis"]["capped_nonterminal_games"],
        "source_evidence_inventory_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_evidence_bytes_and_mtimes_unchanged": True,
        "physical_copy_evidence_bytes_and_mtimes_unchanged": True,
        "frozen_execution_evidence_bytes_and_mtimes_unchanged": True,
    }
    receipt_path = original_output / "physical-copy-verification.json"
    sequence = 1
    while receipt_path.exists():
        receipt_path = original_output / (
            f"physical-copy-verification-supersession-{sequence:02d}.json"
        )
        sequence += 1
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, default=None)
    parser.add_argument("--frozen-execution-root", type=Path, default=None)
    parser.add_argument("--frozen-hidden-root", type=Path, default=None)
    args = parser.parse_args()
    result = verify_copy(
        args.root,
        args.copy,
        args.hidden_root,
        args.python,
        tablebase_path=args.tablebase,
        frozen_execution_root=args.frozen_execution_root,
        frozen_hidden_root=args.frozen_hidden_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
