"""Freeze recovery provenance and continue the update-432 arena screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import (
    run_seed542_expanded_policy_strength_screen as runner,
    seed542_expanded_policy_strength_screen as protocol,
    seed542_update432_strength_screen as experiment,
    verify_seed542_expanded_policy_strength_screen as replay,
    verify_seed542_update432_strength_screen as independent,
)

ROOT = experiment.ROOT
OUT = ROOT / "docs/data/seed542-update-432-strength-screen-v3"
REGISTRATION = "continuation-registration-v4.json"
FREEZE = "continuation-execution-freeze-v4.json"
REGISTRATION_V5 = "continuation-registration-v5.json"
FREEZE_V5 = "continuation-execution-freeze-v5.json"
REGISTRATION_V6 = "continuation-registration-v6.json"
FREEZE_V6 = "continuation-execution-freeze-v6.json"
SOURCES = (
    "ml/alphazero_lite/continue_seed542_update432_strength_screen.py",
    "ml/alphazero_lite/verify_seed542_update432_strength_screen.py",
    "ml/alphazero_lite/test_seed542_update432_strength_screen.py",
    "ml/alphazero_lite/test_continue_seed542_update432_strength_screen.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_hashes(root: Path) -> dict[str, str]:
    return {relative: sha(root / relative) for relative in SOURCES}


def prepare(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    registration, freeze = runner._assert_frozen(root, output)
    if (output / f"{REGISTRATION}").exists() or (output / f"{FREEZE}").exists():
        raise FileExistsError("update432_continuation_v4_already_frozen")
    if (output / "outcomes.jsonl").exists():
        raise ValueError("update432_continuation_requires_empty_committed_ledger")

    old_auth = replay._authenticate_composition
    replay._authenticate_composition = independent.authenticate_composition
    try:
        recovered = replay.validate_pair(
            root, output, 0, require_completion_receipt=False
        )
    finally:
        replay._authenticate_composition = old_auth
    if len(recovered) != 2 or {row["challenger_player"] for row in recovered} != {0, 1}:
        raise ValueError("update432_recovery_pair_not_both_seats")

    raw_paths = {
        name: output / "batches" / f"{name}-0000.{suffix}"
        for name, suffix in (
            ("games", "jsonl"),
            ("report", "json"),
            ("command", "json"),
            ("opening", "jsonl"),
            ("attempt", "json"),
        )
    }
    registration = json.loads((output / "registration.json").read_text())
    command_preflight = sha(output / "command-preflight.jsonl")
    source_hashes = _source_hashes(root)
    command = [
        "systemd-run",
        "--user",
        "--unit=seed542-update432-strength-screen-continuation-v4",
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
        "ml.alphazero_lite.continue_seed542_update432_strength_screen",
        "--execute",
        "--root",
        str(root),
        "--out",
        str(output),
    ]
    continuation = {
        "schema": "seed542_update432_strength_screen_continuation_v4",
        "status": "prospectively_frozen_before_recovery_and_remaining_openings",
        "parent_registration_sha256": sha(output / "registration.json"),
        "parent_execution_freeze_sha256": sha(output / "execution-freeze.json"),
        "parent_command_preflight_sha256": command_preflight,
        "candidate_checkpoint_sha256": sha(
            root / "docs/data/seed542-training-budget-432/D-update-432.npz"
        ),
        "suite_sha256": freeze["suite_sha256"],
        "validated_raw_pair_zero": {
            name: sha(path) for name, path in raw_paths.items()
        },
        "recovery_pair_outcomes": recovered,
        "remaining_openings": list(range(1, 1024)),
        "execution_source_sha256": source_hashes,
        "required_disclosures": registration["disclosures"],
        "arena_protocol": registration["arena_protocol"],
        "analysis": registration["analysis"],
        "durable_execution_command": command,
    }
    registration_path = output / REGISTRATION
    write_json(registration_path, continuation)
    freeze_record = {
        "schema": "seed542_update432_strength_screen_continuation_freeze_v4",
        "status": continuation["status"],
        "registration_sha256": sha(registration_path),
        "parent_registration_sha256": continuation["parent_registration_sha256"],
        "parent_execution_freeze_sha256": continuation[
            "parent_execution_freeze_sha256"
        ],
        "parent_command_preflight_sha256": command_preflight,
        "candidate_checkpoint_sha256": continuation["candidate_checkpoint_sha256"],
        "suite_sha256": continuation["suite_sha256"],
        "validated_raw_pair_zero": continuation["validated_raw_pair_zero"],
        "execution_source_sha256": source_hashes,
        "durable_execution_command": command,
    }
    write_json(output / FREEZE, freeze_record)
    return {
        "status": freeze_record["status"],
        "recovered_pair_games": len(recovered),
        "remaining_openings": 1023,
        "registration_sha256": sha(registration_path),
        "freeze_sha256": sha(output / FREEZE),
    }


def _verify_continuation_freeze(root: Path, output: Path) -> dict[str, Any]:
    registration_path = output / REGISTRATION
    freeze_path = output / FREEZE
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        registration.get("status")
        != "prospectively_frozen_before_recovery_and_remaining_openings"
        or sha(registration_path) != freeze.get("registration_sha256")
        or sha(output / "registration.json") != freeze.get("parent_registration_sha256")
        or sha(output / "execution-freeze.json")
        != freeze.get("parent_execution_freeze_sha256")
        or sha(output / "command-preflight.jsonl")
        != freeze.get("parent_command_preflight_sha256")
    ):
        raise ValueError("update432_continuation_freeze_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"update432_continuation_source_changed:{relative}")
    for name, relative in (
        ("games", "batches/games-0000.jsonl"),
        ("report", "batches/report-0000.json"),
        ("command", "batches/command-0000.json"),
        ("opening", "batches/opening-0000.jsonl"),
        ("attempt", "batches/attempt-0000.json"),
    ):
        if sha(output / relative) != freeze["validated_raw_pair_zero"][name]:
            raise ValueError(f"update432_recovery_pair_changed:{name}")
    return registration


def execute(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    registration = _verify_continuation_freeze(root, output)
    parent_registration = json.loads((output / "registration.json").read_text())
    if (
        registration["candidate_checkpoint_sha256"]
        != parent_registration["candidate"]["training_checkpoint_sha256"]
    ):
        raise ValueError("update432_continuation_candidate_identity_mismatch")

    old_authenticate = replay._authenticate_composition
    replay._authenticate_composition = independent.authenticate_composition
    try:
        # The pre-existing two-seat output is fully replayed before it is bound
        # by its completion receipt and committed to the ledger.
        recovered = runner.recover_opening_zero(root, output)
        if len(recovered) != 2:
            raise ValueError("update432_recovery_commit_did_not_cover_both_seats")

        original_commit = runner.commit_pair

        def commit_with_receipt(
            current_root: Path, current_output: Path, index: int
        ) -> list[dict[str, Any]]:
            _verify_continuation_freeze(current_root, current_output)
            runner.write_output_completion_receipt(current_root, current_output, index)
            return original_commit(current_root, current_output, index)

        runner.commit_pair = commit_with_receipt
        try:
            return runner.run(root, output)
        finally:
            runner.commit_pair = original_commit
    finally:
        replay._authenticate_composition = old_authenticate


def prepare_v5(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    """Freeze recovery of raw opening 1 plus openings 2..1023."""
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    _parent, parent_freeze = runner._assert_frozen(root, output)
    if (output / REGISTRATION_V5).exists() or (output / FREEZE_V5).exists():
        raise FileExistsError("update432_continuation_v5_already_frozen")
    prior_v4_registration_path = output / REGISTRATION
    prior_v4_freeze_path = output / FREEZE
    prior_v4_registration = json.loads(prior_v4_registration_path.read_text())
    prior_v4_freeze = json.loads(prior_v4_freeze_path.read_text())
    if (
        sha(prior_v4_registration_path) != prior_v4_freeze["registration_sha256"]
        or prior_v4_registration["parent_registration_sha256"]
        != parent_freeze["registration_sha256"]
    ):
        raise ValueError("update432_continuation_v4_parent_record_mismatch")
    ledger = runner.rows(output / "outcomes.jsonl")
    if len(ledger) != 2 or {int(row["opening_index"]) for row in ledger} != {0}:
        raise ValueError("update432_continuation_v5_requires_only_pair0_committed")

    old_authenticate = replay._authenticate_composition
    replay._authenticate_composition = independent.authenticate_composition
    try:
        pair0 = replay.validate_pair(root, output, 0)
        replay.reconcile_pair_ledger(0, pair0, ledger)
        raw_pair1 = replay.validate_pair(
            root, output, 1, require_completion_receipt=False
        )
    finally:
        replay._authenticate_composition = old_authenticate
    if len(raw_pair1) != 2 or {row["challenger_player"] for row in raw_pair1} != {0, 1}:
        raise ValueError("update432_continuation_v5_raw_pair1_incomplete")

    evidence = {
        name: sha(output / "batches" / f"{name}-0001.{suffix}")
        for name, suffix in (
            ("games", "jsonl"),
            ("report", "json"),
            ("command", "json"),
            ("opening", "jsonl"),
            ("attempt", "json"),
        )
    }
    sources = _source_hashes(root)
    parent_registration = json.loads((output / "registration.json").read_text())
    durable = [
        "systemd-run",
        "--user",
        "--unit=seed542-update432-strength-screen-continuation-v5",
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
        "ml.alphazero_lite.continue_seed542_update432_strength_screen",
        "--execute-v5",
        "--root",
        str(root),
        "--out",
        str(output),
    ]
    continuation = {
        "schema": "seed542_update432_strength_screen_continuation_v5",
        "status": "prospectively_frozen_before_pair1_recovery_and_openings2_through1023",
        "parent_registration_sha256": sha(output / "registration.json"),
        "parent_execution_freeze_sha256": sha(output / "execution-freeze.json"),
        "prior_v4_registration_sha256": sha(prior_v4_registration_path),
        "prior_v4_freeze_sha256": sha(prior_v4_freeze_path),
        "parent_command_preflight_sha256": sha(output / "command-preflight.jsonl"),
        "candidate_checkpoint_sha256": sha(
            root / "docs/data/seed542-training-budget-432/D-update-432.npz"
        ),
        "suite_sha256": parent_freeze["suite_sha256"],
        "committed_pair0_ledger_sha256": sha(output / "outcomes.jsonl"),
        "raw_pair1_sha256": evidence,
        "raw_pair1_replay": raw_pair1,
        "remaining_openings": list(range(2, 1024)),
        "execution_source_sha256": sources,
        "required_disclosures": parent_registration["disclosures"],
        "arena_protocol": parent_registration["arena_protocol"],
        "analysis": parent_registration["analysis"],
        "durable_execution_command": durable,
    }
    registration_path = output / REGISTRATION_V5
    write_json(registration_path, continuation)
    freeze = {
        "schema": "seed542_update432_strength_screen_continuation_freeze_v5",
        "status": continuation["status"],
        "registration_sha256": sha(registration_path),
        "parent_registration_sha256": continuation["parent_registration_sha256"],
        "parent_execution_freeze_sha256": continuation[
            "parent_execution_freeze_sha256"
        ],
        "prior_v4_registration_sha256": continuation["prior_v4_registration_sha256"],
        "prior_v4_freeze_sha256": continuation["prior_v4_freeze_sha256"],
        "parent_command_preflight_sha256": continuation[
            "parent_command_preflight_sha256"
        ],
        "candidate_checkpoint_sha256": continuation["candidate_checkpoint_sha256"],
        "suite_sha256": continuation["suite_sha256"],
        "committed_pair0_ledger_sha256": continuation["committed_pair0_ledger_sha256"],
        "raw_pair1_sha256": evidence,
        "execution_source_sha256": sources,
        "durable_execution_command": durable,
    }
    freeze_path = output / FREEZE_V5
    write_json(freeze_path, freeze)
    return {
        "status": freeze["status"],
        "validated_pair0_games": len(pair0),
        "validated_recoverable_pair1_games": len(raw_pair1),
        "remaining_openings": 1022,
        "registration_sha256": sha(registration_path),
        "freeze_sha256": sha(freeze_path),
    }


def _verify_v5_freeze(root: Path, output: Path) -> dict[str, Any]:
    registration_path = output / REGISTRATION_V5
    freeze_path = output / FREEZE_V5
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        sha(registration_path) != freeze["registration_sha256"]
        or sha(output / "registration.json") != freeze["parent_registration_sha256"]
        or sha(output / "execution-freeze.json")
        != freeze["parent_execution_freeze_sha256"]
        or sha(output / REGISTRATION) != freeze["prior_v4_registration_sha256"]
        or sha(output / FREEZE) != freeze["prior_v4_freeze_sha256"]
        or sha(output / "command-preflight.jsonl")
        != freeze["parent_command_preflight_sha256"]
        or sha(output / "outcomes.jsonl") != freeze["committed_pair0_ledger_sha256"]
    ):
        raise ValueError("update432_continuation_v5_freeze_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"update432_continuation_v5_source_changed:{relative}")
    for name, suffix in (
        ("games", "jsonl"),
        ("report", "json"),
        ("command", "json"),
        ("opening", "jsonl"),
        ("attempt", "json"),
    ):
        relative = f"batches/{name}-0001.{suffix}"
        if sha(output / relative) != freeze["raw_pair1_sha256"][name]:
            raise ValueError(f"update432_continuation_v5_raw_pair_changed:{name}")
    return registration


def execute_v5(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    _verify_v5_freeze(root, output)
    old_authenticate = replay._authenticate_composition
    old_validate = replay.validate_pair
    replay._authenticate_composition = independent.authenticate_composition

    def validate_with_completion_receipt(
        current_root: Path,
        current_output: Path,
        index: int,
        *,
        require_completion_receipt: bool = True,
    ) -> list[dict[str, Any]]:
        receipt_path = current_output / "batches" / f"completion-{index:04d}.json"
        games_path = current_output / "batches" / f"games-{index:04d}.jsonl"
        report_path = current_output / "batches" / f"report-{index:04d}.json"
        if (
            require_completion_receipt
            and not receipt_path.exists()
            and games_path.is_file()
            and report_path.is_file()
        ):
            _verify_v5_freeze(current_root, current_output)
            runner.write_output_completion_receipt(current_root, current_output, index)
        return old_validate(
            current_root,
            current_output,
            index,
            require_completion_receipt=require_completion_receipt,
        )

    replay.validate_pair = validate_with_completion_receipt
    try:
        pair1 = old_validate(root, output, 1, require_completion_receipt=False)
        if len(pair1) != 2:
            raise ValueError("update432_continuation_v5_pair1_replay_failed")
        runner.write_output_completion_receipt(root, output, 1)
        pair1 = validate_with_completion_receipt(root, output, 1)
        if len(pair1) != 2:
            raise ValueError("update432_continuation_v5_pair1_receipt_failed")
        original_commit = runner.commit_pair

        def commit_with_frozen_receipt(
            current_root: Path, current_output: Path, index: int
        ) -> list[dict[str, Any]]:
            _verify_v5_freeze(current_root, current_output)
            return original_commit(current_root, current_output, index)

        runner.commit_pair = commit_with_frozen_receipt
        try:
            return runner.run(root, output)
        finally:
            runner.commit_pair = original_commit
    finally:
        replay.validate_pair = old_validate
        replay._authenticate_composition = old_authenticate


def _ledger_prefix_hash(rows: list[dict[str, Any]], count: int) -> str:
    payload = b"".join(canonical(row) + b"\n" for row in rows[:count])
    return hashlib.sha256(payload).hexdigest()


def prepare_v6(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    """Freeze recovery of raw opening 2 and the remaining screen."""
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    runner._assert_frozen(root, output)
    v5_registration_path = output / REGISTRATION_V5
    v5_freeze_path = output / FREEZE_V5
    v5_freeze = json.loads(v5_freeze_path.read_text())
    if (
        sha(v5_registration_path) != v5_freeze["registration_sha256"]
        or sha(output / "registration.json") != v5_freeze["parent_registration_sha256"]
        or sha(output / "execution-freeze.json")
        != v5_freeze["parent_execution_freeze_sha256"]
    ):
        raise ValueError("update432_continuation_v6_v5_parent_binding_mismatch")
    if (output / REGISTRATION_V6).exists() or (output / FREEZE_V6).exists():
        raise FileExistsError("update432_continuation_v6_already_frozen")
    ledger = runner.rows(output / "outcomes.jsonl")
    committed_indices = {int(row["opening_index"]) for row in ledger}
    if len(ledger) != 4 or committed_indices != {0, 1}:
        raise ValueError("update432_continuation_v6_requires_pairs0_and1")
    for index in (0, 1):
        pair = replay.validate_pair(root, output, index)
        rows_for_index = [row for row in ledger if int(row["opening_index"]) == index]
        replay.reconcile_pair_ledger(index, pair, rows_for_index)

    old_authenticate = replay._authenticate_composition
    replay._authenticate_composition = independent.authenticate_composition
    try:
        raw_pair2 = replay.validate_pair(
            root, output, 2, require_completion_receipt=False
        )
    finally:
        replay._authenticate_composition = old_authenticate
    if len(raw_pair2) != 2 or {row["challenger_player"] for row in raw_pair2} != {0, 1}:
        raise ValueError("update432_continuation_v6_raw_pair2_incomplete")

    evidence = {
        name: sha(output / "batches" / f"{name}-0002.{suffix}")
        for name, suffix in (
            ("games", "jsonl"),
            ("report", "json"),
            ("command", "json"),
            ("opening", "jsonl"),
            ("attempt", "json"),
        )
    }
    source_hashes = _source_hashes(root)
    parent_registration = json.loads((output / "registration.json").read_text())
    durable = [
        "systemd-run",
        "--user",
        "--unit=seed542-update432-strength-screen-continuation-v6",
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
        "ml.alphazero_lite.continue_seed542_update432_strength_screen",
        "--execute-v6",
        "--root",
        str(root),
        "--out",
        str(output),
    ]
    continuation = {
        "schema": "seed542_update432_strength_screen_continuation_v6",
        "status": "prospectively_frozen_before_pair2_recovery_and_openings3_through1023",
        "parent_registration_sha256": sha(output / "registration.json"),
        "parent_execution_freeze_sha256": sha(output / "execution-freeze.json"),
        "prior_v5_registration_sha256": sha(v5_registration_path),
        "prior_v5_freeze_sha256": sha(v5_freeze_path),
        "parent_command_preflight_sha256": sha(output / "command-preflight.jsonl"),
        "candidate_checkpoint_sha256": sha(
            root / "docs/data/seed542-training-budget-432/D-update-432.npz"
        ),
        "suite_sha256": v5_freeze["suite_sha256"],
        "committed_ledger_prefix_count": len(ledger),
        "committed_ledger_prefix_sha256": sha(output / "outcomes.jsonl"),
        "raw_pair2_sha256": evidence,
        "raw_pair2_replay": raw_pair2,
        "remaining_openings": list(range(3, 1024)),
        "execution_source_sha256": source_hashes,
        "required_disclosures": parent_registration["disclosures"],
        "arena_protocol": parent_registration["arena_protocol"],
        "analysis": parent_registration["analysis"],
        "durable_execution_command": durable,
    }
    registration_path = output / REGISTRATION_V6
    write_json(registration_path, continuation)
    freeze = {
        "schema": "seed542_update432_strength_screen_continuation_freeze_v6",
        "status": continuation["status"],
        "registration_sha256": sha(registration_path),
        "parent_registration_sha256": continuation["parent_registration_sha256"],
        "parent_execution_freeze_sha256": continuation[
            "parent_execution_freeze_sha256"
        ],
        "prior_v5_registration_sha256": continuation["prior_v5_registration_sha256"],
        "prior_v5_freeze_sha256": continuation["prior_v5_freeze_sha256"],
        "parent_command_preflight_sha256": continuation[
            "parent_command_preflight_sha256"
        ],
        "candidate_checkpoint_sha256": continuation["candidate_checkpoint_sha256"],
        "suite_sha256": continuation["suite_sha256"],
        "committed_ledger_prefix_count": len(ledger),
        "committed_ledger_prefix_sha256": continuation[
            "committed_ledger_prefix_sha256"
        ],
        "raw_pair2_sha256": evidence,
        "execution_source_sha256": source_hashes,
        "durable_execution_command": durable,
    }
    freeze_path = output / FREEZE_V6
    write_json(freeze_path, freeze)
    return {
        "status": freeze["status"],
        "validated_committed_games": len(ledger),
        "validated_recoverable_pair2_games": len(raw_pair2),
        "remaining_openings": 1021,
        "registration_sha256": sha(registration_path),
        "freeze_sha256": sha(freeze_path),
    }


def _verify_v6_freeze(root: Path, output: Path) -> dict[str, Any]:
    registration_path = output / REGISTRATION_V6
    freeze_path = output / FREEZE_V6
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        sha(registration_path) != freeze["registration_sha256"]
        or sha(output / "registration.json") != freeze["parent_registration_sha256"]
        or sha(output / "execution-freeze.json")
        != freeze["parent_execution_freeze_sha256"]
        or sha(output / REGISTRATION_V5) != freeze["prior_v5_registration_sha256"]
        or sha(output / FREEZE_V5) != freeze["prior_v5_freeze_sha256"]
        or sha(output / "command-preflight.jsonl")
        != freeze["parent_command_preflight_sha256"]
    ):
        raise ValueError("update432_continuation_v6_freeze_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"update432_continuation_v6_source_changed:{relative}")
    ledger = runner.rows(output / "outcomes.jsonl")
    prefix_count = freeze["committed_ledger_prefix_count"]
    if (
        len(ledger) < prefix_count
        or _ledger_prefix_hash(ledger, prefix_count)
        != freeze["committed_ledger_prefix_sha256"]
    ):
        raise ValueError("update432_continuation_v6_ledger_prefix_changed")
    for name, suffix in (
        ("games", "jsonl"),
        ("report", "json"),
        ("command", "json"),
        ("opening", "jsonl"),
        ("attempt", "json"),
    ):
        relative = f"batches/{name}-0002.{suffix}"
        if sha(output / relative) != freeze["raw_pair2_sha256"][name]:
            raise ValueError(f"update432_continuation_v6_raw_pair_changed:{name}")
    return registration


def execute_v6(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    _verify_v6_freeze(root, output)
    old_authenticate = replay._authenticate_composition
    old_validate = replay.validate_pair
    replay._authenticate_composition = independent.authenticate_composition

    def validate_with_completion_receipt(
        current_root: Path,
        current_output: Path,
        index: int,
        *,
        require_completion_receipt: bool = True,
    ) -> list[dict[str, Any]]:
        receipt_path = current_output / "batches" / f"completion-{index:04d}.json"
        games_path = current_output / "batches" / f"games-{index:04d}.jsonl"
        report_path = current_output / "batches" / f"report-{index:04d}.json"
        if (
            require_completion_receipt
            and not receipt_path.exists()
            and games_path.is_file()
            and report_path.is_file()
        ):
            _verify_v6_freeze(current_root, current_output)
            runner.write_output_completion_receipt(current_root, current_output, index)
        return old_validate(
            current_root,
            current_output,
            index,
            require_completion_receipt=require_completion_receipt,
        )

    replay.validate_pair = validate_with_completion_receipt
    try:
        raw_pair2 = old_validate(root, output, 2, require_completion_receipt=False)
        if len(raw_pair2) != 2:
            raise ValueError("update432_continuation_v6_pair2_replay_failed")
        runner.write_output_completion_receipt(root, output, 2)
        if len(validate_with_completion_receipt(root, output, 2)) != 2:
            raise ValueError("update432_continuation_v6_pair2_receipt_failed")
        original_commit = runner.commit_pair

        def commit_with_prefix_binding(
            current_root: Path, current_output: Path, index: int
        ) -> list[dict[str, Any]]:
            _verify_v6_freeze(current_root, current_output)
            return original_commit(current_root, current_output, index)

        runner.commit_pair = commit_with_prefix_binding
        try:
            return runner.run(root, output)
        finally:
            runner.commit_pair = original_commit
    finally:
        replay.validate_pair = old_validate
        replay._authenticate_composition = old_authenticate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--prepare-v5", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--execute-v5", action="store_true")
    modes.add_argument("--prepare-v6", action="store_true")
    modes.add_argument("--execute-v6", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.root, args.out)
    elif args.prepare_v5:
        result = prepare_v5(args.root, args.out)
    elif args.prepare_v6:
        result = prepare_v6(args.root, args.out)
    elif args.execute_v6:
        result = execute_v6(args.root, args.out)
    elif args.execute_v5:
        result = execute_v5(args.root, args.out)
    else:
        result = execute(args.root, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
