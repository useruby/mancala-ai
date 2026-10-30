"""Recover validated opening 0 and run the frozen continuation 1..1023."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import (
    run_seed542_expanded_policy_strength_screen as runner,
    seed542_expanded_policy_strength_screen as protocol,
    verify_seed542_expanded_policy_strength_screen as verifier,
)

ROOT = runner.ROOT
OUT = ROOT / protocol.OUT
CONT_REG = "continuation-registration-v2.json"
CONT_FREEZE = "continuation-execution-freeze-v2.json"
CONT_COMMANDS = "continuation-command-preflight-v2.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes(root: Path) -> dict[str, str]:
    sources = (
        *runner.RUNTIME_SOURCES,
        "ml/alphazero_lite/continue_seed542_expanded_policy_strength_screen.py",
    )
    return {relative: sha(root / relative) for relative in sources}


def verify_initial_source_snapshots(
    output: Path, original_freeze: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = output / "initial-source-snapshot-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("original_execution_freeze_sha256") != sha(
        output / "execution-freeze.json"
    ):
        raise ValueError("initial_source_snapshot_freeze_binding_mismatch")
    if set(manifest.get("sources", {})) != set(
        original_freeze["execution_source_sha256"]
    ):
        raise ValueError("initial_source_snapshot_coverage_mismatch")
    for relative, expected in original_freeze["execution_source_sha256"].items():
        record = manifest["sources"][relative]
        path = output / record["snapshot_path"]
        if record["sha256"] != expected or sha(path) != expected:
            raise ValueError(f"initial_source_snapshot_hash_mismatch:{relative}")
    return manifest


def create_output_receipt(root: Path, output: Path, index: int) -> Path:
    return runner.write_output_completion_receipt(root, output, index)


def prepare_opening0_recovery(root: Path, output: Path) -> dict[str, Any]:
    """Validate the saved pair and bind recovery code/evidence before commit."""
    root, output = root.resolve(), output.resolve()
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    if sha(output / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("opening0_recovery_original_registration_changed")
    verify_initial_source_snapshots(output, freeze)
    verifier._validate_training_receipt(root, output, registration)
    verifier._authenticate_composition(root, output, registration)
    openings, report = protocol.import_seed511_suite(root)
    if (
        sha(output / "openings.jsonl") != freeze["suite_sha256"]
        or report["suite_sha256"] != freeze["suite_sha256"]
        or openings[0]["opening_index"] != 0
    ):
        raise ValueError("opening0_recovery_suite_binding_mismatch")
    raw_pair = verifier.validate_pair(root, output, 0, require_completion_receipt=False)
    if len(raw_pair) != 2 or {int(row["challenger_player"]) for row in raw_pair} != {
        0,
        1,
    }:
        raise ValueError("opening0_recovery_pair_shape_invalid")
    receipt_path = create_output_receipt(root, output, 0)
    validated_pair = verifier.validate_pair(root, output, 0)
    if validated_pair != raw_pair:
        raise ValueError("opening0_completion_receipt_replay_difference")
    binding = {
        "schema": "seed542_opening0_recovery_binding_v1",
        "status": "opening0_pair_fully_validated_before_ledger_commit",
        "opening_index": 0,
        "original_registration_sha256": sha(output / "registration.json"),
        "original_freeze_sha256": sha(output / "execution-freeze.json"),
        "suite_sha256": sha(output / "openings.jsonl"),
        "completion_receipt_sha256": sha(receipt_path),
        "games_sha256": sha(output / "batches/games-0000.jsonl"),
        "report_sha256": sha(output / "batches/report-0000.json"),
        "training_verification_v2_sha256": sha(
            output / "training-verification-v2.json"
        ),
        "execution_source_sha256": source_hashes(root),
        "normalized_pair": validated_pair,
    }
    _save_json(output / "opening0-recovery-binding.json", binding)
    return binding


def recover_opening_zero(root: Path, output: Path) -> list[dict[str, Any]]:
    """Replay and commit the saved opening-0 pair exactly once."""
    root, output = root.resolve(), output.resolve()
    binding_path = output / "opening0-recovery-binding.json"
    binding = json.loads(binding_path.read_text())
    if binding.get(
        "status"
    ) != "opening0_pair_fully_validated_before_ledger_commit" or binding.get(
        "execution_source_sha256"
    ) != source_hashes(root):
        raise ValueError("opening0_recovery_binding_or_source_mismatch")
    for key, relative in (
        ("original_registration_sha256", "registration.json"),
        ("original_freeze_sha256", "execution-freeze.json"),
        ("suite_sha256", "openings.jsonl"),
        ("completion_receipt_sha256", "batches/completion-0000.json"),
        ("games_sha256", "batches/games-0000.jsonl"),
        ("report_sha256", "batches/report-0000.json"),
        ("training_verification_v2_sha256", "training-verification-v2.json"),
    ):
        if sha(output / relative) != binding[key]:
            raise ValueError(f"opening0_recovery_evidence_changed:{relative}")
    original_registration = json.loads((output / "registration.json").read_text())
    original_freeze = json.loads((output / "execution-freeze.json").read_text())
    if (
        sha(output / "registration.json") != original_freeze["registration_sha256"]
        or sha(output / "openings.jsonl") != original_freeze["suite_sha256"]
        or original_registration.get("execution_root") != str(root)
    ):
        raise ValueError("opening0_original_registration_freeze_mismatch")
    verify_initial_source_snapshots(output, original_freeze)
    training_receipt = json.loads(
        (output / "training-verification-v2.json").read_text()
    )
    if training_receipt.get("status") != "both_seed542_v2_verifiers_passed":
        raise ValueError("opening0_training_verification_v2_incomplete")
    if (
        sha(output / "batches/games-0000.jsonl")
        != "e707d85c3e43e397999616ea805d2a3a5b8fde403d114b743a960cbaff726cd2"
    ):
        raise ValueError("opening0_saved_games_changed")
    if (
        sha(output / "batches/report-0000.json")
        != "b92575de1882a5ea1b57789960f6e5f33335ec797252cebe5adce2b198676b5f"
    ):
        raise ValueError("opening0_saved_report_changed")
    raw_pair = verifier.validate_pair(root, output, 0, require_completion_receipt=False)
    if len(raw_pair) != 2 or {row["challenger_player"] for row in raw_pair} != {0, 1}:
        raise ValueError("opening0_full_seat_pair_not_valid")
    create_output_receipt(root, output, 0)
    validated_pair = verifier.validate_pair(root, output, 0)
    if len(validated_pair) != 2:
        raise ValueError("opening0_completion_receipt_validation_failed")
    if validated_pair != binding["normalized_pair"]:
        raise ValueError("opening0_recovery_pair_changed_after_binding")
    ledger_path = output / "outcomes.jsonl"
    if ledger_path.exists() and ledger_path.stat().st_size:
        pair = verifier.validate_pair(root, output, 0)
        stored = rows(ledger_path)
        if len(stored) != 2:
            raise ValueError("opening0_recovery_ledger_not_two_rows")
        verifier.reconcile_pair_ledger(0, pair, stored)
        if rows(output / "opening0-ledger.jsonl") != stored:
            raise ValueError("opening0_separate_ledger_receipt_mismatch")
        return pair
    payload = "".join(canonical(row).decode() + "\n" for row in validated_pair)
    temporary = ledger_path.with_suffix(".jsonl.pending")
    temporary.write_text(payload)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, ledger_path)
    opening0_ledger = output / "opening0-ledger.jsonl"
    opening0_ledger.write_text(payload)
    return validated_pair


def recover_saved_opening1(root: Path, output: Path) -> list[dict[str, Any]]:
    """Validate and commit opening 1's complete raw v2-continuation pair."""
    root, output = root.resolve(), output.resolve()
    cont_reg_path = output / "continuation-registration-v2.json"
    cont_freeze_path = output / "continuation-execution-freeze-v2.json"
    cont_freeze = json.loads(cont_freeze_path.read_text())
    snapshot_manifest = json.loads(
        (output / "continuation-v2-source-snapshot-manifest.json").read_text()
    )
    if (
        sha(cont_reg_path) != cont_freeze["registration_sha256"]
        or sha(cont_freeze_path) != snapshot_manifest["continuation_freeze_sha256"]
        or snapshot_manifest.get("continuation_freeze_sha256") != sha(cont_freeze_path)
    ):
        raise ValueError("opening1_v2_continuation_freeze_binding_mismatch")
    if set(snapshot_manifest["sources"]) != set(cont_freeze["execution_source_sha256"]):
        raise ValueError("opening1_v2_source_snapshot_coverage_mismatch")
    for relative, expected in cont_freeze["execution_source_sha256"].items():
        record = snapshot_manifest["sources"][relative]
        if (
            record["sha256"] != expected
            or sha(output / record["snapshot_path"]) != expected
        ):
            raise ValueError(f"opening1_v2_frozen_source_snapshot_mismatch:{relative}")

    raw_pair = verifier.validate_pair(root, output, 1, require_completion_receipt=False)
    if len(raw_pair) != 2 or {int(row["challenger_player"]) for row in raw_pair} != {
        0,
        1,
    }:
        raise ValueError("opening1_recovery_requires_complete_swapped_seats")
    receipt_path = runner.write_output_completion_receipt(
        root,
        output,
        1,
        continuation_freeze_sha256=sha(cont_freeze_path),
    )
    validated_pair = verifier.validate_pair(root, output, 1)
    before_receipt = [
        {key: value for key, value in row.items() if key != "completion_receipt_sha256"}
        for row in raw_pair
    ]
    after_receipt = [
        {key: value for key, value in row.items() if key != "completion_receipt_sha256"}
        for row in validated_pair
    ]
    if after_receipt != before_receipt:
        raise ValueError("opening1_completion_receipt_replay_difference")
    binding = {
        "schema": "seed542_opening1_recovery_binding_v1",
        "status": "opening1_pair_fully_validated_before_ledger_commit",
        "opening_index": 1,
        "v2_continuation_registration_sha256": sha(cont_reg_path),
        "v2_continuation_freeze_sha256": sha(cont_freeze_path),
        "v2_source_snapshot_manifest_sha256": sha(
            output / "continuation-v2-source-snapshot-manifest.json"
        ),
        "command_sha256": sha(output / "batches/command-0001.json"),
        "completion_receipt_sha256": sha(receipt_path),
        "games_sha256": sha(output / "batches/games-0001.jsonl"),
        "report_sha256": sha(output / "batches/report-0001.json"),
        "normalized_pair": validated_pair,
    }
    _save_json(output / "opening1-recovery-binding-v1.json", binding)
    ledger_path = output / "outcomes.jsonl"
    ledger = rows(ledger_path)
    if any(int(row["opening_index"]) == 1 for row in ledger):
        raise ValueError("opening1_already_committed")
    if len(ledger) != 2 or {int(row["opening_index"]) for row in ledger} != {0}:
        raise ValueError("opening1_recovery_requires_opening0_prefix")
    updated = ledger + validated_pair
    temporary = ledger_path.with_suffix(".jsonl.pending")
    temporary.write_text("".join(canonical(row).decode() + "\n" for row in updated))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, ledger_path)
    (output / "opening1-ledger.jsonl").write_text(
        "".join(canonical(row).decode() + "\n" for row in validated_pair)
    )
    return validated_pair


def _authenticated_inputs(root: Path, output: Path) -> dict[str, str]:
    original_freeze = json.loads((output / "execution-freeze.json").read_text())
    authenticated = dict(original_freeze["authenticated_file_sha256"])
    snapshot_manifest = verify_initial_source_snapshots(output, original_freeze)
    for relative, record in snapshot_manifest["sources"].items():
        snapshot = output / record["snapshot_path"]
        authenticated[str(snapshot.relative_to(root))] = sha(snapshot)
    for relative, expected in original_freeze["execution_source_sha256"].items():
        current = sha(root / relative)
        # The original freeze's source bytes are authenticated by the separate
        # initial-source-snapshot-manifest. Continuation executable bytes are
        # bound to this new source map; historical snapshots never authorize them.
        authenticated[relative] = current
    for relative, expected in source_hashes(root).items():
        authenticated[relative] = sha(root / relative)
    for relative in (
        "training-verification-v2.json",
        "initial-source-snapshot-manifest.json",
        "opening0-recovery-binding.json",
        "batches/completion-0000.json",
        "opening0-ledger.jsonl",
        "batches/command-0000.json",
        "batches/attempt-0000.json",
        "batches/games-0000.jsonl",
        "batches/report-0000.json",
        "PRELAUNCH-FAILURE-0000.json",
    ):
        path = output / relative
        authenticated[str(path.relative_to(root))] = sha(path)
    return authenticated


def prepare_continuation(root: Path, output: Path) -> dict[str, Any]:
    """Freeze new executable bytes and commands only for openings 1..1023."""
    root, output = root.resolve(), output.resolve()
    original_registration = json.loads((output / "registration.json").read_text())
    original_freeze = json.loads((output / "execution-freeze.json").read_text())
    if sha(output / "registration.json") != original_freeze["registration_sha256"]:
        raise ValueError("continuation_original_registration_changed")
    verify_initial_source_snapshots(output, original_freeze)
    ledger_path = output / "opening0-ledger.jsonl"
    if not ledger_path.is_file() or len(rows(ledger_path)) != 2:
        raise ValueError("continuation_requires_committed_opening0")
    ledger_rows = rows(ledger_path)
    if ledger_rows != rows(output / "outcomes.jsonl"):
        raise ValueError("continuation_opening0_ledger_mismatch")
    continuation_sources = source_hashes(root)
    authenticated = _authenticated_inputs(root, output)
    durable_command = [
        "systemd-run",
        "--user",
        "--unit=seed542-expanded-policy-strength-continuation-v2",
        "--collect",
        f"--property=WorkingDirectory={root}",
        "--property=CPUQuota=25%",
        "--property=CPUWeight=1",
        "--property=MemoryMax=4294967296",
        "--property=AllowedCPUs=30",
        "--property=Nice=19",
        "--property=LimitAS=17179869184",
        f"--setenv=PYTHONPATH={root}",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=BLAS_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--",
        sys.executable,
        "-m",
        "ml.alphazero_lite.continue_seed542_expanded_policy_strength_screen",
        "--root",
        str(root),
        "--out",
        str(output),
        "--execute",
    ]
    continuation_registration = {
        "schema": "seed542_expanded_policy_strength_continuation_v2",
        "status": "prospective_frozen_before_openings_1_through_1023",
        "execution_root": str(root),
        "original_registration_sha256": sha(output / "registration.json"),
        "original_freeze_sha256": sha(output / "execution-freeze.json"),
        "opening0_completion_receipt_sha256": sha(
            output / "batches/completion-0000.json"
        ),
        "opening0_ledger_sha256": sha(ledger_path),
        "training_verification_v2_sha256": sha(
            output / "training-verification-v2.json"
        ),
        "suite_sha256": original_freeze["suite_sha256"],
        "remaining_openings": [*range(1, 1024)],
        "component_assignments": original_registration["component_assignments"],
        "component_hashes": original_registration["component_hashes"],
        "runtime_assets": original_registration["runtime_assets"],
        "runtime": original_registration["runtime"],
        "durable_execution_command": durable_command,
        "arena_protocol": original_registration["arena_protocol"],
        "analysis": original_registration["analysis"],
        "execution_source_sha256": continuation_sources,
        "authenticated_file_sha256": authenticated,
    }
    registration_path = output / CONT_REG
    _save_json(registration_path, continuation_registration)
    registration_sha = sha(registration_path)
    preflight_path = output / CONT_COMMANDS
    commands = []
    for index in range(1, 1024):
        commands.append(
            {
                "opening_index": index,
                "argv": runner.parser_checked_command(root, index),
                "original_registration_sha256": continuation_registration[
                    "original_registration_sha256"
                ],
                "original_suite_sha256": continuation_registration["suite_sha256"],
                "continuation_registration_sha256": registration_sha,
                "component_assignments": continuation_registration[
                    "component_assignments"
                ],
                "component_hashes": continuation_registration["component_hashes"],
                "runtime": continuation_registration["runtime"],
                "runtime_source_sha256": continuation_sources,
            }
        )
    preflight_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in commands
        )
    )
    freeze = {
        "schema": "seed542_expanded_policy_strength_continuation_freeze_v2",
        "status": "frozen_before_openings_1_through_1023",
        "registration_sha256": registration_sha,
        "original_registration_sha256": continuation_registration[
            "original_registration_sha256"
        ],
        "original_freeze_sha256": continuation_registration["original_freeze_sha256"],
        "suite_sha256": continuation_registration["suite_sha256"],
        "opening0_completion_receipt_sha256": continuation_registration[
            "opening0_completion_receipt_sha256"
        ],
        "opening0_ledger_sha256": continuation_registration["opening0_ledger_sha256"],
        "command_preflight_sha256": sha(preflight_path),
        "execution_source_sha256": continuation_sources,
        "authenticated_file_sha256": authenticated,
        "arena_protocol": continuation_registration["arena_protocol"],
        "analysis": continuation_registration["analysis"],
        "durable_execution_command": durable_command,
    }
    _save_json(output / CONT_FREEZE, freeze)
    return {
        "status": freeze["status"],
        "registration_sha256": registration_sha,
        "freeze_sha256": sha(output / CONT_FREEZE),
        "parser_validated_remaining_commands": len(commands),
    }


def prepare_continuation_v3(root: Path, output: Path) -> dict[str, Any]:
    """Freeze the remaining 2..1023 schedule after preserving v1/v2 evidence."""
    root, output = root.resolve(), output.resolve()
    original_registration = json.loads((output / "registration.json").read_text())
    original_freeze = json.loads((output / "execution-freeze.json").read_text())
    v2_registration_path = output / "continuation-registration-v2.json"
    v2_freeze_path = output / "continuation-execution-freeze-v2.json"
    v2_freeze = json.loads(v2_freeze_path.read_text())
    snapshot_manifest_path = output / "continuation-v2-source-snapshot-manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    if (
        sha(output / "registration.json") != original_freeze["registration_sha256"]
        or sha(output / "execution-freeze.json") != v2_freeze["original_freeze_sha256"]
        or sha(v2_registration_path) != v2_freeze["registration_sha256"]
        or sha(v2_freeze_path) != snapshot_manifest["continuation_freeze_sha256"]
        or snapshot_manifest["continuation_freeze_sha256"] != sha(v2_freeze_path)
    ):
        raise ValueError("v3_continuation_parent_freeze_mismatch")
    verify_initial_source_snapshots(output, original_freeze)
    for relative, digest in v2_freeze["execution_source_sha256"].items():
        record = snapshot_manifest["sources"].get(relative)
        if (
            record is None
            or record["sha256"] != digest
            or sha(output / record["snapshot_path"]) != digest
        ):
            raise ValueError(f"v3_v2_execution_snapshot_mismatch:{relative}")

    ledger = rows(output / "outcomes.jsonl")
    prefix = rows(output / "opening0-ledger.jsonl") + rows(
        output / "opening1-ledger.jsonl"
    )
    if len(prefix) != 4 or ledger[:4] != prefix:
        raise ValueError("v3_continuation_requires_validated_openings_0_and_1")
    verifier.reconcile_pair_ledger(
        0,
        verifier.validate_pair(root, output, 0),
        rows(output / "opening0-ledger.jsonl"),
    )
    verifier.reconcile_pair_ledger(
        1,
        verifier.validate_pair(root, output, 1),
        rows(output / "opening1-ledger.jsonl"),
    )

    sources = source_hashes(root)
    authenticated = _authenticated_inputs(root, output)
    for record in snapshot_manifest["sources"].values():
        snapshot = output / record["snapshot_path"]
        authenticated[str(snapshot.relative_to(root))] = sha(snapshot)
    extra_paths = (
        "continuation-registration-v2.json",
        "continuation-execution-freeze-v2.json",
        "continuation-command-preflight-v2.jsonl",
        "continuation-v2-source-snapshot-manifest.json",
        "opening1-recovery-binding-v1.json",
        "opening1-ledger.jsonl",
        "batches/completion-0001.json",
        "batches/command-0001.json",
        "batches/attempt-0001.json",
        "batches/games-0001.jsonl",
        "batches/report-0001.json",
        "EXECUTION-STATUS.md",
        "RECOVERY-STATUS.md",
    )
    for relative in extra_paths:
        authenticated[f"{protocol.OUT}/{relative}"] = sha(output / relative)

    durable_command = [
        "systemd-run",
        "--user",
        "--unit=seed542-expanded-policy-strength-continuation-v3",
        "--collect",
        f"--property=WorkingDirectory={root}",
        "--property=CPUQuota=25%",
        "--property=CPUWeight=1",
        "--property=MemoryMax=4294967296",
        "--property=AllowedCPUs=30",
        "--property=Nice=19",
        "--property=LimitAS=17179869184",
        f"--setenv=PYTHONPATH={root}",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=BLAS_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--",
        sys.executable,
        "-m",
        "ml.alphazero_lite.continue_seed542_expanded_policy_strength_screen",
        "--root",
        str(root),
        "--out",
        str(output),
        "--execute",
        "--continuation-version",
        "3",
    ]
    registration = {
        "schema": "seed542_expanded_policy_strength_continuation_v3",
        "status": "prospective_frozen_before_openings_2_through_1023",
        "execution_root": str(root),
        "original_registration_sha256": sha(output / "registration.json"),
        "original_freeze_sha256": sha(output / "execution-freeze.json"),
        "parent_continuation_registration_sha256": sha(v2_registration_path),
        "parent_continuation_freeze_sha256": sha(v2_freeze_path),
        "opening0_completion_receipt_sha256": sha(
            output / "batches/completion-0000.json"
        ),
        "opening1_completion_receipt_sha256": sha(
            output / "batches/completion-0001.json"
        ),
        "opening0_ledger_sha256": sha(output / "opening0-ledger.jsonl"),
        "opening1_ledger_sha256": sha(output / "opening1-ledger.jsonl"),
        "prefix_ledger_sha256": sha(output / "outcomes.jsonl"),
        "training_verification_v2_sha256": sha(
            output / "training-verification-v2.json"
        ),
        "suite_sha256": original_freeze["suite_sha256"],
        "remaining_openings": list(range(2, 1024)),
        "component_assignments": original_registration["component_assignments"],
        "component_hashes": original_registration["component_hashes"],
        "runtime_assets": original_registration["runtime_assets"],
        "runtime": original_registration["runtime"],
        "durable_execution_command": durable_command,
        "arena_protocol": original_registration["arena_protocol"],
        "analysis": original_registration["analysis"],
        "execution_source_sha256": sources,
        "authenticated_file_sha256": authenticated,
    }
    registration_path = output / "continuation-registration-v3.json"
    _save_json(registration_path, registration)
    registration_sha = sha(registration_path)
    preflight_path = output / "continuation-command-preflight-v3.jsonl"
    command_rows = []
    for index in range(2, 1024):
        command_rows.append(
            {
                "opening_index": index,
                "argv": runner.parser_checked_command(root, index),
                "original_registration_sha256": registration[
                    "original_registration_sha256"
                ],
                "original_suite_sha256": registration["suite_sha256"],
                "continuation_registration_sha256": registration_sha,
                "continuation_version": 3,
                "component_assignments": registration["component_assignments"],
                "component_hashes": registration["component_hashes"],
                "runtime": registration["runtime"],
                "runtime_source_sha256": sources,
            }
        )
    preflight_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in command_rows
        )
    )
    freeze = {
        "schema": "seed542_expanded_policy_strength_continuation_freeze_v3",
        "status": "frozen_before_openings_2_through_1023",
        "registration_sha256": registration_sha,
        "original_registration_sha256": registration["original_registration_sha256"],
        "original_freeze_sha256": registration["original_freeze_sha256"],
        "parent_continuation_freeze_sha256": registration[
            "parent_continuation_freeze_sha256"
        ],
        "suite_sha256": registration["suite_sha256"],
        "opening0_completion_receipt_sha256": registration[
            "opening0_completion_receipt_sha256"
        ],
        "opening1_completion_receipt_sha256": registration[
            "opening1_completion_receipt_sha256"
        ],
        "opening0_ledger_sha256": registration["opening0_ledger_sha256"],
        "opening1_ledger_sha256": registration["opening1_ledger_sha256"],
        "command_preflight_sha256": sha(preflight_path),
        "execution_source_sha256": sources,
        "authenticated_file_sha256": authenticated,
        "arena_protocol": registration["arena_protocol"],
        "analysis": registration["analysis"],
        "durable_execution_command": durable_command,
    }
    freeze_path = output / "continuation-execution-freeze-v3.json"
    _save_json(freeze_path, freeze)
    return {
        "status": freeze["status"],
        "registration_sha256": registration_sha,
        "freeze_sha256": sha(freeze_path),
        "parser_validated_remaining_commands": len(command_rows),
    }


def _assert_continuation_frozen(
    root: Path, output: Path, version: int = 2
) -> tuple[dict[str, Any], dict[str, Any]]:
    registration_path = output / f"continuation-registration-v{version}.json"
    freeze_path = output / f"continuation-execution-freeze-v{version}.json"
    command_preflight_path = output / f"continuation-command-preflight-v{version}.jsonl"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    original_freeze_path = output / "execution-freeze.json"
    original_registration_path = output / "registration.json"
    if (
        sha(registration_path) != freeze["registration_sha256"]
        or sha(original_registration_path) != freeze["original_registration_sha256"]
        or sha(original_freeze_path) != freeze["original_freeze_sha256"]
        or sha(command_preflight_path) != freeze["command_preflight_sha256"]
        or sha(output / "batches/completion-0000.json")
        != freeze["opening0_completion_receipt_sha256"]
        or sha(output / "opening0-ledger.jsonl") != freeze["opening0_ledger_sha256"]
    ):
        raise ValueError("continuation_freeze_binding_mismatch")
    if version >= 3 and (
        sha(output / "continuation-execution-freeze-v2.json")
        != freeze["parent_continuation_freeze_sha256"]
        or sha(output / "batches/completion-0001.json")
        != freeze["opening1_completion_receipt_sha256"]
        or sha(output / "opening1-ledger.jsonl") != freeze["opening1_ledger_sha256"]
    ):
        raise ValueError("continuation_v2_parent_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"continuation_source_changed:{relative}")
    for relative, expected in freeze["authenticated_file_sha256"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if sha(path) != expected:
            raise ValueError(f"continuation_input_changed:{relative}")
    return registration, freeze


def _append_pair_ledger(
    output: Path, pair: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ledger_path = output / "outcomes.jsonl"
    existing = rows(ledger_path) if ledger_path.exists() else []
    index = int(pair[0]["opening_index"])
    if any(int(row["opening_index"]) == index for row in existing):
        raise ValueError(f"continuation_pair_already_committed:{index}")
    if len(pair) != 2 or {int(row["challenger_player"]) for row in pair} != {0, 1}:
        raise ValueError(f"continuation_pair_shape_invalid:{index}")
    updated = existing + pair
    temporary = ledger_path.with_suffix(".jsonl.pending")
    temporary.write_text("".join(canonical(row).decode() + "\n" for row in updated))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, ledger_path)
    return updated


def continue_execution(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    runner._restore_interrupted_tablebase_materialization(root)
    registration, freeze = _assert_continuation_frozen(root, output)
    ledger_path = output / "outcomes.jsonl"
    ledger = rows(ledger_path) if ledger_path.exists() else []
    initial = rows(output / "opening0-ledger.jsonl")
    if ledger[:2] != initial:
        raise ValueError("continuation_opening0_ledger_prefix_mismatch")
    committed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    for index in committed:
        pair = verifier.validate_pair(root, output, index)
        verifier.reconcile_pair_ledger(index, pair, committed[index])

    preflight = {int(row["opening_index"]): row for row in rows(output / CONT_COMMANDS)}
    if set(preflight) != set(range(1, 1024)):
        raise ValueError("continuation_command_preflight_coverage")
    for index in range(1, 1024):
        action = runner.recovery_action(output, index, set(committed))
        batch = output / "batches"
        command_path = batch / f"command-{index:04d}.json"
        attempt_path = batch / f"attempt-{index:04d}.json"
        if action == "skip":
            continue
        if action == "recover":
            verifier.validate_pair(
                root, output, index, require_completion_receipt=False
            )
            if not (batch / f"completion-{index:04d}.json").exists():
                runner.write_output_completion_receipt(
                    root,
                    output,
                    index,
                    continuation_freeze_sha256=sha(output / CONT_FREEZE),
                )
            pair = verifier.validate_pair(root, output, index)
        else:
            argv = runner.parser_checked_command(root, index)
            registered_command = preflight[index]
            if (
                argv != registered_command.get("argv")
                or registered_command.get("continuation_registration_sha256")
                != freeze["registration_sha256"]
                or registered_command.get("runtime_source_sha256")
                != freeze["execution_source_sha256"]
                or registered_command.get("component_assignments")
                != registration["component_assignments"]
                or registered_command.get("component_hashes")
                != registration["component_hashes"]
                or registered_command.get("runtime") != registration["runtime"]
            ):
                raise ValueError(f"continuation_preflight_command_mismatch:{index}")
            command = {
                "opening_index": index,
                "argv": argv,
                "registration_sha256": freeze["original_registration_sha256"],
                "suite_sha256": freeze["suite_sha256"],
                "continuation_registration_sha256": freeze["registration_sha256"],
                "continuation_freeze_sha256": sha(output / CONT_FREEZE),
                "component_assignments": registration["component_assignments"],
                "component_hashes": registration["component_hashes"],
                "runtime": registration["runtime"],
                "runtime_source_sha256": registration["execution_source_sha256"],
                "opening_sha256": sha(batch / f"opening-{index:04d}.jsonl"),
            }
            command_path.write_bytes(canonical(command) + b"\n")
            _append_jsonl(output / "commands.jsonl", command)
            runner._save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "command_sha256": sha(command_path),
                },
            )
            historical = json.loads(
                (root / protocol.SEED511 / "registration.json").read_text()
            )
            historical["runtime_assets"]["exact_root_tablebase"][
                "runtime_materialized_path"
            ] = str(runner.TABLEBASE)
            pointer = runner.seed511_runtime.materialize_bound_tablebase(
                root, historical
            )
            try:
                if (
                    sha(root / "model-artifact/runtime/kalah_v1_21.kvtb")
                    != registration["runtime_assets"]["tablebase_sha256"]
                ):
                    raise ValueError(
                        "continuation_materialized_tablebase_hash_mismatch"
                    )
                result = subprocess.run(argv, cwd=root, check=False)
            finally:
                runner.seed511_runtime.restore_bound_tablebase(
                    root, historical, pointer
                )
            runner._save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "returncode": result.returncode,
                    "command_sha256": sha(command_path),
                },
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"continuation_arena_command_failed:{index}:{result.returncode}"
                )
            verifier.validate_pair(
                root, output, index, require_completion_receipt=False
            )
            runner.write_output_completion_receipt(
                root,
                output,
                index,
                continuation_freeze_sha256=sha(output / CONT_FREEZE),
            )
            pair = verifier.validate_pair(root, output, index)
        _assert_continuation_frozen(root, output)
        pair = verifier.validate_pair(root, output, index)
        ledger = _append_pair_ledger(output, pair)
        committed[index] = pair
        print(
            json.dumps(
                {"opening_index": index, "committed_games": len(ledger)}, sort_keys=True
            ),
            flush=True,
        )

    if len(ledger) != 2048:
        return {
            "status": "incomplete",
            "validated_games": len(ledger),
            "validated_pairs": len(committed),
        }
    result, matrix = runner.runner_analysis(ledger)
    runner._save_json(output / "results.json", result)
    (output / "per-opening-results.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in matrix
        )
    )
    return {
        "status": "all_pairs_committed",
        "validated_games": len(ledger),
        "validated_pairs": len(committed),
        "results": result,
    }


def continue_execution_v3(root: Path, output: Path) -> dict[str, Any]:
    """Execute only 2..1023 under the separately frozen v3 continuation."""
    root, output = root.resolve(), output.resolve()
    runner._restore_interrupted_tablebase_materialization(root)
    registration, freeze = _assert_continuation_frozen(root, output, version=3)
    ledger_path = output / "outcomes.jsonl"
    ledger = rows(ledger_path) if ledger_path.exists() else []
    prefix_ledgers = [
        rows(output / "opening0-ledger.jsonl"),
        rows(output / "opening1-ledger.jsonl"),
    ]
    prefix = [row for part in prefix_ledgers for row in part]
    if ledger[:4] != prefix:
        raise ValueError("continuation_v3_prior_ledger_prefix_mismatch")
    prefix_bytes = "".join(canonical(row).decode() + "\n" for row in prefix).encode()
    if hashlib.sha256(prefix_bytes).hexdigest() != registration["prefix_ledger_sha256"]:
        raise ValueError("continuation_v3_prior_ledger_hash_mismatch")

    committed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    for index, pair_rows in committed.items():
        pair = verifier.validate_pair(root, output, index)
        verifier.reconcile_pair_ledger(index, pair, pair_rows)

    preflight = {
        int(row["opening_index"]): row
        for row in rows(output / "continuation-command-preflight-v3.jsonl")
    }
    if set(preflight) != set(range(2, 1024)):
        raise ValueError("continuation_v3_command_preflight_coverage")

    for index in range(2, 1024):
        action = runner.recovery_action(output, index, set(committed))
        batch = output / "batches"
        command_path = batch / f"command-{index:04d}.json"
        attempt_path = batch / f"attempt-{index:04d}.json"
        if action == "skip":
            continue
        if action == "recover":
            verifier.validate_pair(
                root, output, index, require_completion_receipt=False
            )
            if not (batch / f"completion-{index:04d}.json").exists():
                runner.write_output_completion_receipt(
                    root,
                    output,
                    index,
                    continuation_freeze_sha256=sha(
                        output / "continuation-execution-freeze-v3.json"
                    ),
                )
            pair = verifier.validate_pair(root, output, index)
        else:
            argv = runner.parser_checked_command(root, index)
            preflight_row = preflight[index]
            if (
                argv != preflight_row.get("argv")
                or preflight_row.get("continuation_registration_sha256")
                != freeze["registration_sha256"]
                or preflight_row.get("continuation_version") != 3
                or preflight_row.get("runtime_source_sha256")
                != freeze["execution_source_sha256"]
                or preflight_row.get("component_assignments")
                != registration["component_assignments"]
                or preflight_row.get("component_hashes")
                != registration["component_hashes"]
                or preflight_row.get("runtime") != registration["runtime"]
            ):
                raise ValueError(f"continuation_v3_preflight_command_mismatch:{index}")
            command = {
                "opening_index": index,
                "argv": argv,
                "registration_sha256": freeze["original_registration_sha256"],
                "suite_sha256": freeze["suite_sha256"],
                "continuation_registration_sha256": freeze["registration_sha256"],
                "continuation_freeze_sha256": sha(
                    output / "continuation-execution-freeze-v3.json"
                ),
                "continuation_version": 3,
                "component_assignments": registration["component_assignments"],
                "component_hashes": registration["component_hashes"],
                "runtime": registration["runtime"],
                "runtime_source_sha256": registration["execution_source_sha256"],
                "opening_sha256": sha(batch / f"opening-{index:04d}.jsonl"),
            }
            command_path.write_bytes(canonical(command) + b"\n")
            _append_jsonl(output / "commands.jsonl", command)
            runner._save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "command_sha256": sha(command_path),
                },
            )
            historical = json.loads(
                (root / protocol.SEED511 / "registration.json").read_text()
            )
            historical["runtime_assets"]["exact_root_tablebase"][
                "runtime_materialized_path"
            ] = str(runner.TABLEBASE)
            pointer = runner.seed511_runtime.materialize_bound_tablebase(
                root, historical
            )
            try:
                if (
                    sha(root / "model-artifact/runtime/kalah_v1_21.kvtb")
                    != registration["runtime_assets"]["tablebase_sha256"]
                ):
                    raise ValueError(
                        "continuation_v3_tablebase_materialization_mismatch"
                    )
                result = subprocess.run(argv, cwd=root, check=False)
            finally:
                runner.seed511_runtime.restore_bound_tablebase(
                    root, historical, pointer
                )
            runner._save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "returncode": result.returncode,
                    "command_sha256": sha(command_path),
                },
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"continuation_v3_arena_command_failed:{index}:{result.returncode}"
                )
            verifier.validate_pair(
                root, output, index, require_completion_receipt=False
            )
            runner.write_output_completion_receipt(
                root,
                output,
                index,
                continuation_freeze_sha256=sha(
                    output / "continuation-execution-freeze-v3.json"
                ),
            )
            pair = verifier.validate_pair(root, output, index)
        _assert_continuation_frozen(root, output, version=3)
        pair = verifier.validate_pair(root, output, index)
        ledger = _append_pair_ledger(output, pair)
        committed[index] = pair
        print(
            json.dumps(
                {"opening_index": index, "committed_games": len(ledger)},
                sort_keys=True,
            ),
            flush=True,
        )

    if len(ledger) != 2048:
        return {
            "status": "incomplete",
            "validated_games": len(ledger),
            "validated_pairs": len(committed),
        }
    result, matrix = runner.runner_analysis(ledger)
    runner._save_json(output / "results.json", result)
    (output / "per-opening-results.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in matrix
        )
    )
    return {
        "status": "all_pairs_committed",
        "validated_games": len(ledger),
        "validated_pairs": len(committed),
        "results": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--recover-opening-0", action="store_true")
    parser.add_argument("--recover-opening-1", action="store_true")
    parser.add_argument("--prepare-opening-0-recovery", action="store_true")
    parser.add_argument("--freeze-continuation", action="store_true")
    parser.add_argument("--freeze-continuation-v3", action="store_true")
    parser.add_argument("--continuation-version", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_opening_0_recovery:
        result = prepare_opening0_recovery(args.root, args.out)
    elif args.recover_opening_0:
        result = recover_opening_zero(args.root, args.out)
        result = {"status": "opening0_recovered_and_committed", "games": len(result)}
    elif args.recover_opening_1:
        pair = recover_saved_opening1(args.root, args.out)
        result = {"status": "opening1_recovered_and_committed", "games": len(pair)}
    elif args.freeze_continuation:
        result = prepare_continuation(args.root, args.out)
    elif args.freeze_continuation_v3:
        result = prepare_continuation_v3(args.root, args.out)
    elif args.execute:
        if args.continuation_version >= 3:
            result = continue_execution_v3(args.root, args.out)
        else:
            result = continue_execution(args.root, args.out)
    else:
        raise SystemExit(
            "choose --prepare-opening-0-recovery, --recover-opening-0, "
            "--recover-opening-1, --freeze-continuation, "
            "--freeze-continuation-v3, or --execute"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
