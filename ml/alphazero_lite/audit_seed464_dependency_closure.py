"""Audit seed464's immutable input and source dependency closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = Path("docs/data/seed464-historical-value-removal")
AUDIT = Path("docs/data/seed464-dependency-closure-audit")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency(
    root: Path,
    registration: dict[str, Any],
    path: str,
    *,
    role: str,
    registered_source_path: str | None = None,
    expected_hash: str | None = None,
    enforcement: str | None = None,
    notes: str,
) -> dict[str, Any]:
    file_path = root / path
    actual = sha(file_path) if file_path.is_file() else None
    source_hash = registration.get("source_sha256", {}).get(registered_source_path)
    direct_input = registration.get("input_sha256", {}).get(path)
    return {
        "path": path,
        "role": role,
        "registered_source_path": registered_source_path,
        "registered_source_sha256": source_hash,
        "direct_frozen_input_sha256": direct_input,
        "frozen_expected_input_sha256": expected_hash,
        "actual_sha256": actual,
        "actual_matches_expected": actual == expected_hash if expected_hash else None,
        "actual_matches_direct_input": actual == direct_input if direct_input else None,
        "enforcement_location": enforcement,
        "notes": notes,
    }


def build_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    attempt = root / ATTEMPT
    registration = json.loads((attempt / "registration.json").read_text())
    registration_sha = sha(attempt / "registration.json")
    loader_path = "ml/alphazero_lite/seed435_adam_direction.py"
    loader_hash = registration["source_sha256"][loader_path]
    seed416_path = "docs/data/seed416-policy-target-softening/registration-v3.json"
    seed416_hash = "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066"
    receipt_specs = (
        (
            "evaluation receipt",
            "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
            "fbcaff1e183b2247673dd7f301c410e11524d13bbe47e067079ee8947aa60ee1",
            "seed435_adam_direction.py:33-40,109-111",
        ),
        (
            "supplemental receipt",
            "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
            "1dd0f8e2442a32654b29c8969c1300cbbfecb83f84809d2a478290e0bda9209e",
            "seed435_adam_direction.py:41-45,109-111",
        ),
    )
    rows = [
        dependency(
            root,
            registration,
            seed416_path,
            role="input read by seed435._inputs",
            registered_source_path=loader_path,
            expected_hash=seed416_hash,
            enforcement="seed435_adam_direction.py:98-103",
            notes="Not directly listed in seed464 input_sha256. The registered seed435 source contains the immutable expected digest and rejects mismatch.",
        )
    ]
    for role, path, expected, enforcement in receipt_specs:
        rows.append(
            dependency(
                root,
                registration,
                path,
                role=role,
                registered_source_path=loader_path,
                expected_hash=expected,
                enforcement=enforcement,
                notes="Not directly listed in seed464 input_sha256. Expected digest is in the registered seed435 loader and is checked during _inputs.",
            )
        )

    seed416 = json.loads((root / seed416_path).read_text())
    derivative_rows = []
    for replay in seed416["replays"]:
        spec = seed416["derivatives"][replay["name"]]["A"]
        path = str(Path(spec["derivative"]).resolve().relative_to(root))
        derivative_rows.append(
            dependency(
                root,
                registration,
                path,
                role=f"replay derivative:{replay['name']}",
                registered_source_path=loader_path,
                expected_hash=spec["derivative_sha256"],
                enforcement=None,
                notes="seed435._inputs passes this absolute derivative path to train.load_jsonl_replay but never compares its bytes to derivative_sha256. The seed464 freeze binds the five canonical source JSONL.gz files instead, not these consumed derivative bytes.",
            )
        )

    helper_paths = (
        "ml/alphazero_lite/verify_seed434_census_source.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/seed432_policy_target_census.py",
        "ml/alphazero_lite/seed433_census_correction.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
        "ml/alphazero_lite/build_opening_suite.py",
        "ml/alphazero_lite/kalah_rules.py",
    )
    seed462 = json.loads(
        (
            root / "docs/data/seed462-historical-policy-removal/registration.json"
        ).read_text()
    )
    seed416_source_hashes = seed416.get("source_hashes", {})
    helper_rows = []
    for path in helper_paths:
        expected_462 = seed462.get("source_sha256", {}).get(path)
        expected_416 = seed416_source_hashes.get(path)
        expected = expected_462 or expected_416
        via = (
            "seed462 registration.json, itself directly input-hash-bound by seed464"
            if expected_462
            else "seed416 registration-v3 source_hashes, whose file SHA is enforced by registered seed435 source"
            if expected_416
            else None
        )
        actual = sha(root / path)
        helper_rows.append(
            {
                "path": path,
                "actual_sha256": actual,
                "seed464_registered_source_sha256": registration["source_sha256"].get(
                    path
                ),
                "transitive_expected_sha256": expected,
                "transitive_commitment": via,
                "actual_matches_expected": actual == expected if expected else None,
                "runtime_enforcement": None,
                "notes": "Imported during seed435._inputs source reconstruction. seed464 does not verify the current imported module file against this nested commitment before execution."
                if expected
                else "Imported by the census reconstruction path and has no SHA commitment in seed464, seed462, or seed416 registration source inventories.",
            }
        )

    # Audit all direct input bindings without changing the frozen registration.
    direct_inputs = [
        dependency(
            root,
            registration,
            path,
            role="directly registered seed464 input",
            expected_hash=expected,
            enforcement="seed464 runner execute-time input_sha256 check",
            notes="Directly covered by the frozen registration.",
        )
        for path, expected in sorted(registration["input_sha256"].items())
    ]
    endpoint = json.loads((attempt / "endpoint-evidence.json").read_text())
    receipt = json.loads((attempt / "receipt.json").read_text())
    metrics = {
        "arms": endpoint["arms"],
        "effects_T_minus_C": endpoint["effects_T_minus_C"],
        "gates_as_recorded_descriptively": endpoint["gates"],
    }
    # There is no registered decision: the old report is descriptive only because
    # consumed derivative bytes and executable helper identities were not closed.
    missing_enforcement = [row["path"] for row in derivative_rows]
    missing_source_commitments = [
        row["path"] for row in helper_rows if row["transitive_expected_sha256"] is None
    ]
    return {
        "schema": "seed464-dependency-closure-audit-v1",
        "classification": "invalid_attempt_dependency_closure_incomplete",
        "registered_decision": None,
        "original_attempt": {
            "registration_sha256": registration_sha,
            "loader_source_sha256": loader_hash,
            "publication_receipt_sha256": sha(attempt / "receipt.json"),
            "recorded_decision": receipt["decision"]["status"],
            "recorded_decision_interpretation": "descriptive only; not a registered decision",
        },
        "specified_dependencies": rows,
        "consumed_derivatives": derivative_rows,
        "imported_helper_sources": helper_rows,
        "remaining_direct_input_bindings": direct_inputs,
        "descriptive_results": metrics,
        "blocking_findings": {
            "consumed_derivative_hashes_not_enforced": missing_enforcement,
            "imported_helper_sources_without_any_prospective_hash": missing_source_commitments,
            "relocated_bundle_requires_unregistered_seed416_registration_and_two_receipts": True,
        },
        "required_for_separately_authorized_run": [
            "Bind every derivative JSONL byte hash actually consumed, or reconstruct derivatives into the registered bundle and verify exact bytes before loading.",
            "Freeze and hash the complete recursive executable source closure, including verify_seed434_census_source.py and its imported policy/census helpers; check those hashes at runtime.",
            "Include seed416 registration-v3.json and both seed427 receipts in the explicit input inventory and physically relocated bundle, even though seed435 currently transitively commits and verifies their identities.",
            "Re-run the complete registration-to-publication workflow tests before creating a new immutable registration; do not alter this attempt's freeze or regenerate its outputs.",
        ],
    }


def publish(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("audit_output_exists")
    report = build_report(root)
    output.mkdir(parents=True)
    report_path = output / "dependency-closure-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Seed464 dependency-closure audit",
        "",
        f"**Classification:** `{report['classification']}`  ",
        "**Registered decision:** none. The original endpoint gate arithmetic is preserved as descriptive evidence only.",
        "",
        "## Requested dependencies",
        "",
        "| Dependency path | Registered source path / SHA256 | Frozen expected input SHA256 | Actual SHA256 | Enforced by executed loader |",
        "|---|---|---|---|---|",
    ]
    for row in report["specified_dependencies"]:
        lines.append(
            f"| `{row['path']}` | `{row['registered_source_path']}` / `{row['registered_source_sha256']}` | `{row['frozen_expected_input_sha256']}` | `{row['actual_sha256']}` | `{row['enforcement_location']}` |"
        )
    lines.extend(
        [
            "",
            "The named registration and receipt files are transitively bound: the frozen seed435 loader contains their expected SHA256 values, its exact source SHA256 is registered by seed464, and `_inputs()` checks each file. Their actual hashes match. They are absent from seed464's direct input inventory, which explains the original relocated-bundle failure but is not itself an identity failure.",
            "",
            "## Other consumed replay dependencies",
            "",
            "| Derivative path | Frozen expected SHA256 (seed416 registration) | Actual SHA256 | Direct seed464 input binding | Loader digest check |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["consumed_derivatives"]:
        lines.append(
            f"| `{row['path']}` | `{row['frozen_expected_input_sha256']}` | `{row['actual_sha256']}` | `{row['direct_frozen_input_sha256']}` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |"
        )
    lines.extend(
        [
            "",
            "All five current derivative files match the digest recorded in seed416 registration-v3. However, seed435 `_inputs()` does not enforce those derivative digests before passing the absolute derivative paths to `train.load_jsonl_replay`. The five canonical source JSONL.gz hashes in seed464's input inventory do not bind the distinct transformed derivative bytes actually loaded.",
            "",
            "## Executable helper source closure",
            "",
            "| Imported helper source | Prospective expected SHA256 chain | Actual SHA256 | seed464 source hash | Runtime comparison |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["imported_helper_sources"]:
        lines.append(
            f"| `{row['path']}` | `{row['transitive_expected_sha256']}` via {row['transitive_commitment']} | `{row['actual_sha256']}` | `{row['seed464_registered_source_sha256']}` | none |"
        )
    lines.extend(
        [
            "",
            "The census helper imports additional target/census helpers. `seed432_policy_target_census.py`, `seed433_census_correction.py`, and `exact_root_policy_targets.py` have no prospective source hash in seed464, seed462, or seed416 inventories. seed464 also does not check the nested seed462/seed416 source commitments against live imported helper files. This leaves the executed source closure incomplete.",
            "",
            "## Descriptive outcomes (not registered decisions)",
            "",
            "```json",
            json.dumps(report["descriptive_results"], indent=2, sort_keys=True),
            "```",
            "",
            "## Required before a separately authorized run",
            "",
        ]
    )
    lines.extend(
        f"- {item}" for item in report["required_for_separately_authorized_run"]
    )
    (output / "dependency-closure-report.md").write_text("\n".join(lines) + "\n")
    # The receipt records audit code and report bytes. Original attempt artifacts
    # are only read and remain outside the publication write set.
    source_path = Path(__file__).resolve()
    receipt = {
        "schema": "seed464-dependency-closure-receipt-v1",
        "classification": report["classification"],
        "registered_decision": None,
        "audit_source_path": str(source_path.relative_to(root)),
        "audit_source_sha256": sha(source_path),
        "report_sha256": sha(report_path),
        "report_markdown_sha256": sha(output / "dependency-closure-report.md"),
        "original_registration_sha256": report["original_attempt"][
            "registration_sha256"
        ],
        "original_receipt_sha256": report["original_attempt"][
            "publication_receipt_sha256"
        ],
    }
    (output / "dependency-closure-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    report = (
        publish(args.root, args.root / args.output)
        if args.publish
        else build_report(args.root)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
