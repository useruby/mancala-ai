"""Physically copy seed551 evidence and verify it with the source worktree hidden."""

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

SCREEN = "docs/data/seed551-controlled-policy-strength-screen"
DEFAULT_COPY = Path("/home/alex/Mancala/seed551-physical-copy-corrected")
TABLEBASE = Path("/home/alex/Mancala/ai/.tmp/seed542-screen-runtime/kalah_v1_21.kvtb")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_bindings(source: Path, destination: Path) -> set[Path]:
    """Copy precisely the registered source and evidence closure with mtimes."""
    registration = json.loads((source / SCREEN / "registration.json").read_text())
    relative_files = {
        *registration["execution_source_sha256"],
        *registration["authenticated_file_sha256"],
        "pyproject.toml",
    }
    relative_files.update(
        str(Path(SCREEN) / path.relative_to(source / SCREEN))
        for path in (source / SCREEN).rglob("*")
        if path.is_file()
    )
    relative_files.add("model-artifact/current/search_policy.json")
    relative_files.add("model-artifact/runtime/kalah_v1_21.kvtb")
    relative_files.add("model-artifact/runtime/kalah_v1_tablebase")

    evidence: set[Path] = set()
    for relative in sorted(relative_files):
        source_path = source / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"physical_copy_missing_binding:{relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source_path, target)
        if sha(target) != sha(source_path):
            raise ValueError(f"physical_copy_byte_mismatch:{relative}")
        evidence.add(target)
    scratch = destination / ".tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    return evidence


def snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in sorted(paths, key=str)
    }


def verify_copy(
    source: Path,
    destination: Path,
    *,
    outside_cwd: Path,
) -> dict[str, Any]:
    source, destination = source.resolve(), destination.resolve()
    outside_cwd = outside_cwd.resolve()
    if destination.exists():
        raise FileExistsError(f"physical_copy_destination_exists:{destination}")
    evidence = copy_bindings(source, destination)
    evidence.add(TABLEBASE)
    before = snapshot(evidence)
    hidden = source.with_name(source.name + "-hidden-for-physical-verification")
    if hidden.exists():
        raise FileExistsError(f"original_worktree_hidden_path_exists:{hidden}")
    os.chdir(outside_cwd)
    os.rename(source, hidden)
    try:
        if source.exists():
            raise RuntimeError("original_worktree_still_accessible_during_verification")
        command = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed551_controlled_policy_strength",
            "--root",
            str(destination),
            "--out",
            str(destination / SCREEN),
            "--tablebase",
            str(TABLEBASE),
            "--allow-relocated-copy",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(destination)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TMPDIR"] = str(destination / ".tmp")
        completed = subprocess.run(
            command,
            cwd=outside_cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        if (
            report.get("status") != "verified_complete"
            or report.get("verified_games") != 2048
            or report.get("read_only_bytes_and_mtimes_unchanged") is not True
        ):
            raise ValueError("physical_copy_independent_verification_failed")
    finally:
        os.rename(hidden, source)
    after = snapshot(evidence)
    if before != after:
        raise ValueError("physical_copy_verifier_changed_bytes_or_mtimes")
    result = {
        "schema": "seed551_physical_copy_verification_v1",
        "status": "physical_copy_verified_read_only",
        "source_worktree_hidden_during_verification": True,
        "outside_working_directory": str(outside_cwd),
        "physical_copy_root": str(destination),
        "command": command,
        "exit_status": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "verified_games": report["verified_games"],
        "verified_pairs": report["verified_pairs"],
        "capped_nonterminal_games": report["capped_nonterminal_games"],
        "copied_evidence_files": len(evidence) - 1,
        "tablebase_sha256": sha(TABLEBASE),
        "evidence_bytes_and_mtimes_unchanged": True,
        "source_worktree_restored": source.is_dir(),
        "preserved_previous_attempt_failure": json.loads(
            (source / "docs/data/seed551-physical-copy-failure-0001.json").read_text()
        ),
        "preserved_previous_physical_copy_verification": {
            "path": "docs/data/seed551-physical-copy-verification-superseded-v2.json",
            "sha256": sha(
                source
                / "docs/data/seed551-physical-copy-verification-superseded-v2.json"
            ),
            "status": "physical_copy_verified_read_only",
            "superseded_by": "P2 raw-report runtime identity correction",
        },
    }
    if not result["source_worktree_restored"]:
        raise RuntimeError("original_worktree_restore_failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--copy", type=Path, default=DEFAULT_COPY)
    parser.add_argument("--outside-cwd", type=Path, default=Path("/home/alex/Mancala"))
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    result = verify_copy(args.source, args.copy, outside_cwd=args.outside_cwd)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
