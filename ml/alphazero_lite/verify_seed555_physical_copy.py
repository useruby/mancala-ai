"""Independently verify a physical evidence copy with original paths hidden."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def verify_physical_copy(
    *,
    copy_root: Path,
    copied_tablebase: Path,
    original_root: Path,
    original_tablebase: Path,
    unrelated_cwd: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    copy_root = copy_root.resolve()
    copied_tablebase = copied_tablebase.resolve()
    original_root = original_root.resolve()
    original_tablebase = original_tablebase.resolve()
    unrelated_cwd = unrelated_cwd.resolve()
    receipt_path = receipt_path.resolve()
    hidden_root = original_root.with_name(
        original_root.name + ".physical-verification-hidden"
    )
    hidden_tablebase = original_tablebase.with_name(
        original_tablebase.name + ".physical-verification-hidden"
    )
    if hidden_root.exists() or hidden_tablebase.exists():
        raise FileExistsError("seed555_physical_verification_hidden_path_exists")
    registration = json.loads(
        (
            copy_root / "docs/data/seed555-policy-strength-screen/registration.json"
        ).read_text()
    )
    if registration.get("execution_root") != str(original_root):
        raise ValueError("seed555_physical_copy_original_root_binding_mismatch")
    output = copy_root / "docs/data/seed555-policy-strength-screen"
    child_source = (
        "import hashlib,json; from pathlib import Path; "
        "from ml.alphazero_lite.verify_seed555_policy_strength_screen import verify; "
        f"blocked=[Path({str(original_root)!r}),Path({str(original_tablebase)!r})]; "
        "assert all(not path.exists() for path in blocked), 'original_path_accessible'; "
        f"root=Path({str(copy_root)!r}); out=root/'docs/data/seed555-policy-strength-screen'; "
        f"result=verify(root,out,tablebase_path=Path({str(copied_tablebase)!r}),allow_relocated_copy=True); "
        "summary={key:result[key] for key in ('status','validated_games','validated_pairs','read_only_bytes_and_mtimes_unchanged')}; "
        "summary['decision']=result['analysis']['decision']; "
        "summary['evidence_snapshot_sha256']=hashlib.sha256(json.dumps(result['evidence_snapshot'],sort_keys=True,separators=(',',':')).encode()).hexdigest(); "
        "summary['original_paths_inaccessible']=True; print(json.dumps(summary,sort_keys=True))"
    )
    command = [sys.executable, "-c", child_source]
    env = {**os.environ, "PYTHONPATH": str(copy_root)}
    original_cwd = Path.cwd()
    result: subprocess.CompletedProcess[str] | None = None
    try:
        os.chdir(unrelated_cwd)
        original_root.rename(hidden_root)
        original_tablebase.rename(hidden_tablebase)
        result = subprocess.run(
            command,
            cwd=unrelated_cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if hidden_tablebase.exists():
            hidden_tablebase.rename(original_tablebase)
        if hidden_root.exists():
            hidden_root.rename(original_root)
        os.chdir(original_cwd)
    if result is None or result.returncode != 0:
        raise RuntimeError(
            "seed555_physical_copy_verification_failed:"
            + (result.stderr if result else "child_not_started")
        )
    verification = json.loads(result.stdout)
    if (
        verification.get("status") != "verified_complete"
        or verification.get("validated_games") != 2048
        or verification.get("validated_pairs") != 1024
        or verification.get("read_only_bytes_and_mtimes_unchanged") is not True
        or verification.get("original_paths_inaccessible") is not True
    ):
        raise ValueError("seed555_physical_copy_verification_incomplete")
    receipt = {
        "schema": "seed555_physical_copy_verification_v1",
        "status": "passed",
        "copy_root": str(copy_root),
        "copy_output": str(output),
        "copied_materialized_tablebase": str(copied_tablebase),
        "original_execution_root_hidden_as": str(hidden_root),
        "original_materialized_tablebase_hidden_as": str(hidden_tablebase),
        "working_directory": str(unrelated_cwd),
        "command": command,
        "exit_status": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "verification": verification,
        "copy_bytes_and_mtimes_unchanged": True,
        "tracked_runtime_assets_modified": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(f".{receipt_path.name}.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)
    directory_fd = os.open(
        receipt_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt


def publish_read_only_verification(root: Path, output: Path) -> dict[str, Any]:
    """Publish the independent verifier result and a hash-bound receipt."""
    from ml.alphazero_lite import seed555_policy_strength_screen as protocol
    from ml.alphazero_lite.verify_seed555_policy_strength_screen import verify

    root, output = root.resolve(), output.resolve()
    command = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.verify_seed555_policy_strength_screen",
        "--root",
        str(root),
        "--out",
        str(output),
    ]
    result = verify(root, output)
    if (
        result.get("status") != "verified_complete"
        or result.get("validated_games") != 2048
        or result.get("validated_pairs") != 1024
        or result.get("read_only_bytes_and_mtimes_unchanged") is not True
    ):
        raise ValueError("seed555_independent_verification_incomplete")
    verification_path = output / "verification.json"
    _write_json(verification_path, result)
    files = {
        path.relative_to(output).as_posix(): protocol.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "verification-receipt.json"
    }
    receipt = {
        "schema": "seed555_read_only_verification_receipt_v1",
        "status": "independent_read_only_verified_complete",
        "command": command,
        "exit_status": 0,
        "verification_result": {
            "status": result["status"],
            "validated_games": result["validated_games"],
            "validated_pairs": result["validated_pairs"],
            "read_only_bytes_and_mtimes_unchanged": result[
                "read_only_bytes_and_mtimes_unchanged"
            ],
            "analysis": result["analysis"],
        },
        "verification_sha256": protocol.sha256(verification_path),
        "evidence_sha256": files,
    }
    receipt_path = output / "verification-receipt.json"
    _write_json(receipt_path, receipt)
    return receipt


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-root", type=Path, required=True)
    parser.add_argument("--copied-tablebase", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--original-tablebase", type=Path, required=True)
    parser.add_argument("--unrelated-cwd", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--publish-verification", action="store_true")
    args = parser.parse_args()
    if args.publish_verification:
        result = publish_read_only_verification(args.original_root, args.receipt)
    else:
        result = verify_physical_copy(
            copy_root=args.copy_root,
            copied_tablebase=args.copied_tablebase,
            original_root=args.original_root,
            original_tablebase=args.original_tablebase,
            unrelated_cwd=args.unrelated_cwd,
            receipt_path=args.receipt,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_path": str(args.receipt),
                "verified_games": result.get(
                    "verification", result.get("verification_result", {})
                ).get("validated_games"),
                "verified_pairs": result.get(
                    "verification", result.get("verification_result", {})
                ).get("validated_pairs"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
