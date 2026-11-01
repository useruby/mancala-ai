"""Append-only audit of suite and continuation bindings for the 432 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import (
    seed542_expanded_policy_strength_screen as screen_protocol,
    verify_seed542_update432_strength_screen as independent,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed542-update-432-strength-screen-v3"
PAIR_BINDINGS = "continuation-pair-bindings-v6.jsonl"
CORRECTION_RECEIPT = "verification-correction-v1.json"
REGISTERED_OUTPUT_REL = "docs/data/seed542-update-432-strength-screen-v3"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_continuation_record(
    version: int,
    registration: dict[str, Any],
    freeze: dict[str, Any],
    registration_sha256: str,
) -> None:
    if freeze.get("registration_sha256") != registration_sha256 or registration.get(
        "execution_source_sha256"
    ) != freeze.get("execution_source_sha256"):
        raise ValueError(f"continuation_v{version}_registration_freeze_mismatch")


def outcome_projection(pair: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "opening_index",
        "game_position",
        "challenger_player",
        "winner",
        "challenger_margin",
        "game_sha256",
        "report_sha256",
        "opening_sha256",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in sorted(pair, key=lambda row: int(row["game_position"]))
    ]


def verify_opening_and_command_binding(
    index: int,
    frozen_suite_line: bytes,
    played_opening_bytes: bytes,
    preflight: dict[str, Any],
    command: dict[str, Any],
    *,
    freeze: dict[str, Any],
    registration: dict[str, Any],
    completion_receipt: dict[str, Any] | None = None,
    command_sha256: str | None = None,
    games_sha256: str | None = None,
    report_sha256: str | None = None,
) -> None:
    """Tie one played opening and command to their exact frozen indexed rows."""
    if completion_receipt is not None and (
        completion_receipt.get("opening_index") != index
        or completion_receipt.get("registration_sha256")
        != freeze["registration_sha256"]
        or completion_receipt.get("suite_sha256") != freeze["suite_sha256"]
        or completion_receipt.get("opening_sha256") != sha_bytes(played_opening_bytes)
        or completion_receipt.get("command_sha256") != command_sha256
        or completion_receipt.get("games_sha256") != games_sha256
        or completion_receipt.get("report_sha256") != report_sha256
    ):
        raise ValueError(f"completion_receipt_self_binding_mismatch:{index}")
    if played_opening_bytes != frozen_suite_line:
        raise ValueError(f"opening_not_equal_to_frozen_suite_row:{index}")
    if (
        preflight.get("opening_index") != index
        or preflight.get("registration_sha256") != freeze["registration_sha256"]
        or preflight.get("suite_sha256") != freeze["suite_sha256"]
        or preflight.get("runtime") != registration["runtime"]
        or preflight.get("component_assignments")
        != registration["component_assignments"]
        or preflight.get("component_hashes") != registration["component_hashes"]
        or preflight.get("runtime_source_sha256")
        != registration["execution_source_sha256"]
    ):
        raise ValueError(f"preflight_frozen_index_binding_mismatch:{index}")
    for field in (
        "argv",
        "registration_sha256",
        "suite_sha256",
        "component_assignments",
        "component_hashes",
        "runtime",
        "runtime_source_sha256",
    ):
        if command.get(field) != preflight.get(field):
            raise ValueError(
                f"played_command_differs_from_frozen_preflight:{index}:{field}"
            )
    if (
        command.get("opening_index") != index
        or command.get("opening_sha256") != sha_bytes(played_opening_bytes)
        or command.get("component_assignments") != registration["component_assignments"]
        or command.get("component_hashes") != registration["component_hashes"]
    ):
        raise ValueError(
            f"played_command_opening_or_component_binding_mismatch:{index}"
        )


def verify_continuation_chain(
    root: Path, output: Path, registration: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    """Authenticate continuation v4-v6 registrations, parents, prefixes and sources."""
    records = {}
    paths = {}
    for version in (4, 5, 6):
        registration_path = output / f"continuation-registration-v{version}.json"
        freeze_path = output / f"continuation-execution-freeze-v{version}.json"
        continuation = json.loads(registration_path.read_text())
        continuation_freeze = json.loads(freeze_path.read_text())
        verify_continuation_record(
            version, continuation, continuation_freeze, sha(registration_path)
        )
        if (
            continuation.get("parent_registration_sha256")
            != sha(output / "registration.json")
            or continuation.get("parent_execution_freeze_sha256")
            != sha(output / "execution-freeze.json")
            or continuation.get("candidate_checkpoint_sha256")
            != registration["candidate"]["training_checkpoint_sha256"]
            or continuation.get("suite_sha256") != freeze["suite_sha256"]
        ):
            raise ValueError(f"continuation_v{version}_parent_or_candidate_mismatch")
        if (
            continuation_freeze.get("parent_registration_sha256")
            != continuation.get("parent_registration_sha256")
            or continuation_freeze.get("parent_execution_freeze_sha256")
            != continuation.get("parent_execution_freeze_sha256")
            or continuation_freeze.get("suite_sha256") != freeze["suite_sha256"]
        ):
            raise ValueError(f"continuation_v{version}_freeze_parent_mismatch")
        for relative, expected in continuation["execution_source_sha256"].items():
            if continuation_freeze["execution_source_sha256"].get(relative) != expected:
                raise ValueError(
                    f"continuation_v{version}_source_digest_mismatch:{relative}"
                )
            if (
                relative
                not in {
                    "ml/alphazero_lite/continue_seed542_update432_strength_screen.py",
                    "ml/alphazero_lite/test_continue_seed542_update432_strength_screen.py",
                }
                and sha(root / relative) != expected
            ):
                raise ValueError(
                    f"continuation_v{version}_stable_source_changed:{relative}"
                )
        records[version] = continuation
        paths[version] = (registration_path, freeze_path)

    v4, v5, v6 = records[4], records[5], records[6]
    v4_reg, v4_freeze = paths[4]
    v5_reg, v5_freeze = paths[5]
    v6_reg, v6_freeze = paths[6]
    if (
        v5.get("prior_v4_registration_sha256") != sha(v4_reg)
        or v5.get("prior_v4_freeze_sha256") != sha(v4_freeze)
        or v6.get("prior_v5_registration_sha256") != sha(v5_reg)
        or v6.get("prior_v5_freeze_sha256") != sha(v5_freeze)
        or json.loads(v5_freeze.read_text()).get("prior_v4_registration_sha256")
        != sha(v4_reg)
        or json.loads(v5_freeze.read_text()).get("prior_v4_freeze_sha256")
        != sha(v4_freeze)
        or json.loads(v6_freeze.read_text()).get("prior_v5_registration_sha256")
        != sha(v5_reg)
        or json.loads(v6_freeze.read_text()).get("prior_v5_freeze_sha256")
        != sha(v5_freeze)
    ):
        raise ValueError("continuation_parent_chain_hash_mismatch")
    ledger = rows(output / "outcomes.jsonl")
    if (
        v5.get("committed_pair0_ledger_sha256")
        != sha_bytes(b"".join(canonical(row) + b"\n" for row in ledger[:2]))
        or v6.get("committed_ledger_prefix_count") != 4
    ):
        raise ValueError("continuation_recovery_ledger_binding_mismatch")
    if (
        v4.get("remaining_openings") != list(range(1, 1024))
        or v5.get("remaining_openings") != list(range(2, 1024))
        or v6.get("remaining_openings") != list(range(3, 1024))
    ):
        raise ValueError("continuation_remaining_openings_mismatch")

    # The v6 source is the active continuation implementation. Earlier
    # continuations are append-only ancestors, so authenticate their exact
    # frozen maps and parent-chain digests rather than comparing them to later
    # source revisions.
    current_sources = {
        relative: sha(root / relative) for relative in v6["execution_source_sha256"]
    }
    if current_sources != v6["execution_source_sha256"]:
        raise ValueError("continuation_v6_active_source_bytes_mismatch")

    if outcome_projection(ledger[:2]) != outcome_projection(
        v4["recovery_pair_outcomes"]
    ):
        raise ValueError("continuation_v4_recovery_pair_ledger_mismatch")
    if outcome_projection(ledger[2:4]) != outcome_projection(v5["raw_pair1_replay"]):
        raise ValueError("continuation_v5_recovery_pair_ledger_mismatch")
    if (
        len(ledger) < v6["committed_ledger_prefix_count"]
        or sha_bytes(
            b"".join(
                canonical(row) + b"\n"
                for row in ledger[: v6["committed_ledger_prefix_count"]]
            )
        )
        != v6["committed_ledger_prefix_sha256"]
    ):
        raise ValueError("continuation_v6_committed_ledger_prefix_mismatch")

    for version, index in ((4, 0), (5, 1), (6, 2)):
        cont = records[version]
        raw_hashes = (
            cont.get("validated_raw_pair_zero")
            if version == 4
            else cont.get("raw_pair1_sha256")
            if version == 5
            else cont.get("raw_pair2_sha256")
        )
        if raw_hashes is None:
            raw_hashes = cont.get("validated_raw_pair_zero", {})
        for name, suffix in (
            ("games", "jsonl"),
            ("report", "json"),
            ("command", "json"),
            ("opening", "jsonl"),
            ("attempt", "json"),
        ):
            path = output / "batches" / f"{name}-{index:04d}.{suffix}"
            if sha(path) != raw_hashes.get(name):
                raise ValueError(
                    f"continuation_v{version}_recovery_evidence_mismatch:{name}"
                )
        pair_rows = [
            row
            for row in rows(output / "outcomes.jsonl")
            if int(row["opening_index"]) == index
        ]
        replay_rows = (
            cont.get("recovery_pair_outcomes")
            if version == 4
            else cont.get("raw_pair1_replay")
            if version == 5
            else cont.get("raw_pair2_replay")
        )
        if outcome_projection(pair_rows) != outcome_projection(replay_rows):
            raise ValueError(f"continuation_v{version}_recovery_ledger_mismatch")
    return {
        "versions": [4, 5, 6],
        "parent_links_valid": True,
        "registration_freeze_source_maps_match": True,
        "v6_active_sources_match": True,
        "recovery_pairs_and_ledger_prefixes_bound": True,
    }


def opening_bindings(root: Path, output: Path) -> list[dict[str, Any]]:
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    suite_lines = (output / "openings.jsonl").read_bytes().splitlines(keepends=True)
    preflight_rows = rows(output / "command-preflight.jsonl")
    preflight = {int(row["opening_index"]): row for row in preflight_rows}
    if len(suite_lines) != 1024 or set(preflight) != set(range(1024)):
        raise ValueError("frozen_suite_or_preflight_coverage_mismatch")
    bindings = []
    for index in range(1024):
        batch = output / "batches"
        opening_path = batch / f"opening-{index:04d}.jsonl"
        command_path = batch / f"command-{index:04d}.json"
        games_path = batch / f"games-{index:04d}.jsonl"
        report_path = batch / f"report-{index:04d}.json"
        completion_path = batch / f"completion-{index:04d}.json"
        actual_opening = opening_path.read_bytes()
        command = json.loads(command_path.read_text())
        frozen_row = preflight[index]
        verify_opening_and_command_binding(
            index,
            suite_lines[index],
            actual_opening,
            frozen_row,
            command,
            freeze=freeze,
            registration=registration,
            completion_receipt=json.loads(completion_path.read_text()),
            command_sha256=sha(command_path),
            games_sha256=sha(games_path),
            report_sha256=sha(report_path),
        )
        version = 4 if index == 0 else 5 if index == 1 else 6
        continuation_reg_path = output / f"continuation-registration-v{version}.json"
        continuation_freeze_path = (
            output / f"continuation-execution-freeze-v{version}.json"
        )
        continuation_reg = json.loads(continuation_reg_path.read_text())
        bindings.append(
            {
                "opening_index": index,
                "continuation_version": version,
                "continuation_registration_sha256": sha(continuation_reg_path),
                "continuation_freeze_sha256": sha(continuation_freeze_path),
                "parent_registration_sha256": continuation_reg[
                    "parent_registration_sha256"
                ],
                "parent_execution_freeze_sha256": continuation_reg[
                    "parent_execution_freeze_sha256"
                ],
                "suite_sha256": freeze["suite_sha256"],
                "frozen_suite_opening_sha256": sha_bytes(suite_lines[index]),
                "played_opening_sha256": sha(opening_path),
                "preflight_row_sha256": sha_bytes(canonical(frozen_row)),
                "command_sha256": sha(command_path),
                "games_sha256": sha(games_path),
                "report_sha256": sha(report_path),
                "completion_receipt_sha256": sha(completion_path),
            }
        )
    return bindings


def verify_pair_binding(row: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in (
        "opening_index",
        "continuation_version",
        "continuation_registration_sha256",
        "continuation_freeze_sha256",
        "parent_registration_sha256",
        "parent_execution_freeze_sha256",
        "suite_sha256",
        "frozen_suite_opening_sha256",
        "played_opening_sha256",
        "preflight_row_sha256",
        "command_sha256",
        "games_sha256",
        "report_sha256",
        "completion_receipt_sha256",
    ):
        if row.get(field) != expected.get(field):
            raise ValueError(f"continuation_pair_binding_mismatch:{field}")


def publish_and_verify(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    continuation_receipt = verify_continuation_chain(root, output, registration, freeze)
    bindings = opening_bindings(root, output)
    binding_path = output / PAIR_BINDINGS
    if binding_path.exists():
        existing = rows(binding_path)
        if existing != bindings:
            raise ValueError("existing_continuation_pair_bindings_changed")
    else:
        binding_path.write_text(
            "".join(canonical(row).decode() + "\n" for row in bindings)
        )
    published_bindings = rows(binding_path)
    if len(published_bindings) != 1024:
        raise ValueError("continuation_pair_binding_coverage_mismatch")
    for expected, actual in zip(bindings, published_bindings, strict=True):
        verify_pair_binding(actual, expected)

    original_pair_validator = independent.base_verifier._validate_command_and_report

    def validate_at_registered_path(*args: Any, **kwargs: Any) -> None:
        # A detached physical copy stores raw evidence elsewhere, while commands
        # remain bound to the original execution-root output paths.
        screen_protocol.OUT = REGISTERED_OUTPUT_REL
        original_pair_validator(*args, **kwargs)

    independent.base_verifier._validate_command_and_report = validate_at_registered_path
    try:
        result = independent.verify(root, output)
    finally:
        independent.base_verifier._validate_command_and_report = original_pair_validator
    correction = {
        "schema": "seed542_update432_strength_screen_verification_correction_v1",
        "status": "append_only_verification_correction_passed",
        "original_registration_sha256": sha(output / "registration.json"),
        "original_execution_freeze_sha256": sha(output / "execution-freeze.json"),
        "continuation_chain": continuation_receipt,
        "per_opening_suite_and_command_bindings_verified": 1024,
        "per_opening_continuation_bindings_verified": 1024,
        "pair_bindings_file": PAIR_BINDINGS,
        "pair_bindings_sha256": sha(binding_path),
        "game_replay_and_bootstrap": result,
    }
    write_path = output / CORRECTION_RECEIPT
    if write_path.exists():
        if json.loads(write_path.read_text()) != correction:
            raise ValueError("existing_verification_correction_receipt_changed")
    else:
        write_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(publish_and_verify(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
