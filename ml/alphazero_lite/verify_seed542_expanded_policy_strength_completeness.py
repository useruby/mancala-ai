"""Independent completeness check before the registered result is calculated.

This verifier validates every frozen pair and its receipts, but intentionally
does not aggregate scores, bootstrap, or calculate an advancement decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed542_expanded_policy_strength_screen as suite_protocol
from ml.alphazero_lite import (
    verify_seed542_expanded_policy_strength_screen as pair_verifier,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / suite_protocol.OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in sorted(paths, key=str)
    }


def _evidence_paths(
    root: Path, output: Path, registrations: list[dict[str, Any]]
) -> set[Path]:
    paths = {path for path in output.rglob("*") if path.is_file()}
    for registration in registrations:
        for relative in (
            *registration.get("execution_source_sha256", {}),
            *registration.get("authenticated_file_sha256", {}),
        ):
            path = Path(relative)
            paths.add(path if path.is_absolute() else root / path)
    for manifest_path in (
        output / "initial-source-snapshot-manifest.json",
        output / "continuation-v2-source-snapshot-manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text())
        paths.update(
            output / row["snapshot_path"] for row in manifest["sources"].values()
        )
    return paths


def _verify_frozen_command_preflights(
    output: Path,
    original_freeze: dict[str, Any],
    v2_freeze: dict[str, Any],
    v3_freeze: dict[str, Any],
) -> None:
    """Require every command-preflight file to match its frozen digest."""
    bindings = (
        ("original", "command-preflight.jsonl", original_freeze),
        ("v2", "continuation-command-preflight-v2.jsonl", v2_freeze),
        ("v3", "continuation-command-preflight-v3.jsonl", v3_freeze),
    )
    for label, filename, freeze in bindings:
        path = output / filename
        expected_sha256 = freeze.get("command_preflight_sha256")
        if not expected_sha256 or sha(path) != expected_sha256:
            raise ValueError(
                f"completeness_frozen_command_preflight_hash_mismatch:{label}"
            )


def verify_completeness(root: Path, output: Path) -> dict[str, Any]:
    """Replay all 2,048 games and reconcile every frozen binding, without analysis."""
    root, output = root.resolve(), output.resolve()
    original_registration_path = output / "registration.json"
    original_freeze_path = output / "execution-freeze.json"
    v2_registration_path = output / "continuation-registration-v2.json"
    v2_freeze_path = output / "continuation-execution-freeze-v2.json"
    v3_registration_path = output / "continuation-registration-v3.json"
    v3_freeze_path = output / "continuation-execution-freeze-v3.json"
    original_registration = json.loads(original_registration_path.read_text())
    original_freeze = json.loads(original_freeze_path.read_text())
    v2_registration = json.loads(v2_registration_path.read_text())
    v2_freeze = json.loads(v2_freeze_path.read_text())
    v3_registration = json.loads(v3_registration_path.read_text())
    v3_freeze = json.loads(v3_freeze_path.read_text())

    if (
        sha(original_registration_path) != original_freeze["registration_sha256"]
        or sha(v2_registration_path) != v2_freeze["registration_sha256"]
        or sha(v3_registration_path) != v3_freeze["registration_sha256"]
        or sha(original_freeze_path) != v3_freeze["original_freeze_sha256"]
        or sha(v2_freeze_path) != v3_freeze["parent_continuation_freeze_sha256"]
        or sha(output / "openings.jsonl") != original_freeze["suite_sha256"]
        or sha(output / "batches/completion-0000.json")
        != v3_registration["opening0_completion_receipt_sha256"]
        or sha(output / "batches/completion-0001.json")
        != v3_registration["opening1_completion_receipt_sha256"]
    ):
        raise ValueError("completeness_freeze_or_suite_binding_mismatch")
    _verify_frozen_command_preflights(output, original_freeze, v2_freeze, v3_freeze)
    if (
        v2_registration.get("original_registration_sha256")
        != original_freeze["registration_sha256"]
        or v2_registration.get("opening0_ledger_sha256")
        != sha(output / "opening0-ledger.jsonl")
        or v3_registration.get("opening0_ledger_sha256")
        != sha(output / "opening0-ledger.jsonl")
        or v3_registration.get("opening1_ledger_sha256")
        != sha(output / "opening1-ledger.jsonl")
    ):
        raise ValueError("completeness_prior_pair_binding_mismatch")

    observed = _evidence_paths(
        root, output, [original_registration, v2_registration, v3_registration]
    )
    before = _snapshot(observed)
    if (
        sha(output / "suite-import-report.json")
        != original_registration["suite"]["import_report_sha256"]
        or sha(output / "overlap-summary.json")
        != original_registration["suite"]["overlap_summary_sha256"]
        or sha(output / "overlap-per-opening.jsonl")
        != original_registration["suite"]["overlap_per_opening_sha256"]
    ):
        raise ValueError("completeness_suite_or_overlap_report_hash_mismatch")
    openings, import_report = suite_protocol.import_seed511_suite(root)
    if (
        import_report["suite_sha256"] != original_freeze["suite_sha256"]
        or json.loads((output / "suite-import-report.json").read_text())
        != import_report
    ):
        raise ValueError("completeness_imported_suite_replay_mismatch")
    overlap = suite_protocol.overlap_report(root, openings)
    if (
        json.loads((output / "overlap-summary.json").read_text())
        != {key: value for key, value in overlap.items() if key != "per_opening"}
        or read_rows(output / "overlap-per-opening.jsonl") != overlap["per_opening"]
    ):
        raise ValueError("completeness_overlap_recomputation_mismatch")

    source_manifest = json.loads(
        (output / "initial-source-snapshot-manifest.json").read_text()
    )
    if source_manifest["original_execution_freeze_sha256"] != sha(
        original_freeze_path
    ) or set(source_manifest["sources"]) != set(
        original_freeze["execution_source_sha256"]
    ):
        raise ValueError("completeness_original_source_snapshot_binding_mismatch")
    for relative, digest in original_freeze["execution_source_sha256"].items():
        snapshot = output / source_manifest["sources"][relative]["snapshot_path"]
        if sha(snapshot) != digest:
            raise ValueError(
                f"completeness_original_source_snapshot_mismatch:{relative}"
            )

    v2_manifest = json.loads(
        (output / "continuation-v2-source-snapshot-manifest.json").read_text()
    )
    if v2_manifest["continuation_freeze_sha256"] != sha(v2_freeze_path) or set(
        v2_manifest["sources"]
    ) != set(v2_freeze["execution_source_sha256"]):
        raise ValueError("completeness_v2_source_snapshot_binding_mismatch")
    for relative, digest in v2_freeze["execution_source_sha256"].items():
        if sha(output / v2_manifest["sources"][relative]["snapshot_path"]) != digest:
            raise ValueError(f"completeness_v2_source_snapshot_mismatch:{relative}")

    for relative, digest in v3_freeze["execution_source_sha256"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"completeness_execution_source_changed:{relative}")
    for freeze in (original_freeze, v2_freeze, v3_freeze):
        frozen_sources = set(freeze["execution_source_sha256"])
        for relative, digest in freeze["authenticated_file_sha256"].items():
            if relative in frozen_sources and freeze is not v3_freeze:
                continue
            path = Path(relative)
            if not path.is_absolute():
                path = root / path
            if sha(path) != digest:
                raise ValueError(f"completeness_authenticated_file_changed:{relative}")

    pair_verifier._authenticate_composition(root, output, original_registration)
    pair_verifier._validate_training_receipt(root, output, original_registration)

    ledger = read_rows(output / "outcomes.jsonl")
    committed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    if len(ledger) != 2048 or set(committed) != set(range(1024)):
        raise ValueError("completeness_ledger_coverage_mismatch")
    command_log = {
        int(row["opening_index"]): row for row in read_rows(output / "commands.jsonl")
    }
    if set(command_log) != set(range(1024)):
        raise ValueError("completeness_command_log_coverage_mismatch")
    preflight = {
        0: {
            int(row["opening_index"]): row
            for row in read_rows(output / "command-preflight.jsonl")
        },
        2: {
            int(row["opening_index"]): row
            for row in read_rows(output / "continuation-command-preflight-v3.jsonl")
        },
    }
    v2_preflight = {
        int(row["opening_index"]): row
        for row in read_rows(output / "continuation-command-preflight-v2.jsonl")
    }
    if (
        set(preflight[0]) != set(range(1024))
        or set(v2_preflight) != set(range(1, 1024))
        or set(preflight[2]) != set(range(2, 1024))
    ):
        raise ValueError("completeness_preflight_command_coverage_mismatch")
    for index, opening in enumerate(openings):
        command = json.loads(
            (output / "batches" / f"command-{index:04d}.json").read_text()
        )
        if command_log[index] != command:
            raise ValueError(f"completeness_command_log_receipt_mismatch:{index}")
        expected = (
            preflight[0][index]
            if index == 0
            else v2_preflight[index]
            if index == 1
            else preflight[2][index]
        )
        for key in (
            "argv",
            "component_assignments",
            "component_hashes",
            "runtime",
            "runtime_source_sha256",
        ):
            if command.get(key) != expected.get(key):
                raise ValueError(
                    f"completeness_command_preflight_mismatch:{index}:{key}"
                )
        raw_pair = pair_verifier.validate_pair(root, output, index)
        pair_verifier.reconcile_pair_ledger(index, raw_pair, committed[index])
    if before != _snapshot(observed):
        raise ValueError("completeness_verifier_changed_evidence_bytes_or_mtimes")
    return {
        "status": "all_1024_pairs_and_2048_games_independently_verified",
        "analysis_performed": False,
        "games": len(ledger),
        "pairs": len(committed),
        "openings": len(openings),
        "evidence_files_snapshotted": len(observed),
        "evidence_bytes_and_mtimes_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(
        json.dumps(verify_completeness(args.root, args.out), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
