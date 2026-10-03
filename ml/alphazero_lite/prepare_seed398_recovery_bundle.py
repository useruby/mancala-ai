"""Package and verify the frozen inputs for the seed398 composition study.

This recovery utility copies byte-identical historical inputs into repo-local
``.tmp/seed398-frozen-input-bundle``. It never registers or launches an arena.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ml.alphazero_lite.build_opening_suite import load_suite_jsonl
from ml.alphazero_lite.build_opening_suite import INITIAL_STATE, canonical_key
from ml.alphazero_lite.arena import apply_opening_moves, canonical_game_state_hash
from ml.alphazero_lite.frozen_opponent_identity import (
    validate_frozen_opponent_identity,
)
from ml.alphazero_lite.opening_exclusion_contract import (
    create_manifest,
    historical_opening_identities,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / ".tmp/seed398-frozen-input-bundle"
DATA = ROOT / "docs/data/seed461-e1-e4-corrected-diagnostic"
EXPECTED_E4 = "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4"
EXPECTED_SEED455 = "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inventory(root: Path, rows: list[dict[str, str]]) -> None:
    """Fail closed when an inventoried bundle input is missing or changed."""
    for row in rows:
        path = root / row["bundle_path"]
        if not path.is_file():
            raise ValueError(f"bundle_file_missing:{row['bundle_path']}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"bundle_file_hash_mismatch:{row['bundle_path']}")


def _copy_file(source: Path, relative: str, rows: list[dict[str, str]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"recovery_source_missing:{source}")
    expected = sha256(source)
    previous = next((row for row in rows if row["bundle_path"] == relative), None)
    if previous is not None:
        if previous["sha256"] != expected:
            raise ValueError(f"bundle_duplicate_source_conflict:{relative}")
        return
    target = DESTINATION / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != expected:
        raise ValueError(f"bundle_destination_conflict:{relative}")
    if not target.exists():
        shutil.copyfile(source, target)
    rows.append(
        {
            "source_path": str(source),
            "bundle_path": relative,
            "sha256": expected,
            "bytes": str(source.stat().st_size),
        }
    )


def build_bundle() -> dict[str, Any]:
    registration = json.loads((DATA / "registration.json").read_text())
    binding = json.loads((DATA / "evaluation-binding.json").read_text())
    receipt_path = ROOT / "docs/data/seed461-o0-checkpoint-recovery-provenance.json"
    receipt = json.loads(receipt_path.read_text())
    manifest_path = DATA / "opening-exclusion-manifest.json"
    historical = json.loads(manifest_path.read_text())
    sources = [
        {"path": row["path"], "kind": row["kind"]} for row in historical["sources"]
    ]
    recomputed = create_manifest(sources)
    if recomputed != historical or historical["excluded_state_count"] != 93_366:
        raise ValueError("historical_396_exclusion_manifest_mismatch")

    e4 = ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz"
    e4_artifact = ROOT / ".tmp/seed461-order-confirmation/artifacts/O0-E4"
    opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    seed455 = (
        ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz"
    )
    if sha256(e4) != EXPECTED_E4 or sha256(e4_artifact / "model.npz") != EXPECTED_E4:
        raise ValueError("original_o0_e4_checkpoint_identity_mismatch")
    if sha256(seed455) != EXPECTED_SEED455:
        raise ValueError("seed455_checkpoint_identity_mismatch")
    for filename, expected_hash in binding["artifact_hashes"]["E4"].items():
        if sha256(e4_artifact / filename) != expected_hash:
            raise ValueError(f"e4_runtime_artifact_identity_mismatch:{filename}")
    for filename, expected_hash in (
        ("weights.json", registration["input_binding"]["opponent"]["weights_sha256"]),
        ("metadata.json", registration["input_binding"]["opponent"]["metadata_sha256"]),
        (
            "search_policy.json",
            registration["input_binding"]["opponent"]["sidecar_sha256"],
        ),
    ):
        if sha256(opponent / filename) != expected_hash:
            raise ValueError(f"seed455_runtime_artifact_identity_mismatch:{filename}")
    runtime_contract = resolve_strength_comparison_runtime_contract(
        current_artifact=opponent, challenger_artifact=e4_artifact
    )
    seed455_runtime_contract = resolve_strength_comparison_runtime_contract(
        current_artifact=opponent, challenger_artifact=opponent
    )
    if runtime_contract != seed455_runtime_contract:
        raise ValueError("component_runtime_contract_mismatch")
    validate_frozen_opponent_identity(
        opponent,
        registration["input_binding"]["opponent"],
        runtime_contract,
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fixed_files = [
        (receipt_path, "historical/seed461-o0-checkpoint-recovery-provenance.json"),
        (DATA / "registration.json", "historical/seed461-e1-e4-registration.json"),
        (DATA / "evaluation-binding.json", "historical/seed461-e1-e4-binding.json"),
        (manifest_path, "historical/seed461-e1-e4-exclusion-manifest.json"),
        (
            DATA / "seed397-openings-v2.jsonl",
            "evaluation-suites/seed396-openings-v2.jsonl",
        ),
        (
            ROOT / "docs/data/seed397-o0-e1-e4-openings.jsonl",
            "evaluation-suites/seed397-openings-v2.jsonl",
        ),
        (
            ROOT / "docs/data/seed397-o0-e1-e4-registration.json",
            "historical/seed397-registration.json",
        ),
        (
            ROOT / "docs/data/seed397-o0-e1-e4-evaluation-binding.json",
            "historical/seed397-binding.json",
        ),
        (
            ROOT / "docs/data/seed397-evidence-erratum.md",
            "historical/seed397-erratum.md",
        ),
        (
            ROOT / "docs/data/seed397-evidence-erratum-audit.json",
            "historical/seed397-erratum-audit.json",
        ),
        (e4, "artifacts/original-o0-e4-checkpoint.npz"),
        (seed455, "artifacts/seed455-checkpoint.npz"),
        (
            ROOT / "model-artifact/runtime/kalah_v1_tablebase",
            "runtime/kalah_v1_tablebase",
        ),
        (ROOT / "model-artifact/runtime/kalah_v1_21.kvtb", "runtime/kalah_v1_21.kvtb"),
    ]
    for source, relative in fixed_files:
        _copy_file(source, relative, rows)
    for label, artifact in (
        ("original-o0-e4", e4_artifact),
        ("seed455-opponent", opponent),
    ):
        for filename in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            path = artifact / filename
            if path.is_file():
                _copy_file(path, f"artifacts/{label}/{filename}", rows)
    for source in historical["sources"]:
        origin = Path(source["path"])
        if not origin.is_absolute():
            origin = ROOT / origin
        relative = str(origin.relative_to(ROOT))
        _copy_file(origin, f"exclusion-sources/{relative}", rows)

    # Preserve completed outcome ledgers, not just opening definitions; their
    # consumed state identities can then be audited independently of the suite.
    for which, report_set in (("396", binding["reports"]),):
        for candidate, evidence in report_set.items():
            _copy_file(
                Path(evidence["report"]), f"outcomes/seed{which}/{candidate}.json", rows
            )
            _copy_file(
                Path(evidence["games"]),
                f"outcomes/seed{which}/{candidate}-games.jsonl",
                rows,
            )
    seed397_binding = json.loads(
        (ROOT / "docs/data/seed397-o0-e1-e4-evaluation-binding.json").read_text()
    )
    for candidate, evidence in seed397_binding.get("reports", {}).items():
        for key in ("report", "games"):
            source = Path(evidence[key])
            if not source.is_absolute():
                source = ROOT / source
            if source.is_file():
                _copy_file(
                    source,
                    f"outcomes/seed397/{candidate}{'-games' if key == 'games' else ''}.{'jsonl' if key == 'games' else 'json'}",
                    rows,
                )

    suite_sets: list[set[str]] = []
    for path in (
        DATA / "seed397-openings-v2.jsonl",
        ROOT / "docs/data/seed397-o0-e1-e4-openings.jsonl",
    ):
        declared, actual = historical_opening_identities(load_suite_jsonl(str(path)))
        suite_sets.append(set(declared) | set(actual))
    consumed_sets: dict[str, set[str]] = {"396": set(), "397": set()}
    consumed_sources = {
        "396": [
            ROOT / ".tmp/seed461-e1-e4-corrected-diagnostic/E1-games.jsonl",
            ROOT / ".tmp/seed461-e1-e4-corrected-diagnostic/E4-games.jsonl",
        ],
        "397": [
            ROOT / ".tmp/seed397-o0-e1-e4/E1-games.jsonl",
            ROOT / ".tmp/seed397-o0-e1-e4/E4-games.jsonl",
        ],
    }
    for diagnostic, paths in consumed_sources.items():
        for raw_path in paths:
            _copy_file(
                raw_path,
                f"outcomes/seed{diagnostic}/{raw_path.stem}.jsonl",
                rows,
            )
            rows_for_run = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows_for_run) != 1024:
                raise ValueError(f"completed_game_ledger_count_mismatch:{diagnostic}")
            run_ids: set[str] = set()
            for game_row in rows_for_run:
                applied_count = game_row["opening_applied_prefix_length"]
                prefix = [int(move) for move in game_row["opening_prefix_moves"]]
                if not isinstance(applied_count, int) or not 0 <= applied_count <= len(
                    prefix
                ):
                    raise ValueError(f"invalid_consumed_prefix_length:{diagnostic}")
                game = KalahGame.from_state(INITIAL_STATE)
                applied = apply_opening_moves(game, prefix[:applied_count])
                if (
                    applied != applied_count
                    or canonical_game_state_hash(game) != game_row["opening_state_hash"]
                ):
                    raise ValueError(f"consumed_state_identity_mismatch:{diagnostic}")
                run_ids.add(canonical_key(game.to_state()))
            if len(run_ids) != 512:
                raise ValueError(f"consumed_start_uniqueness_mismatch:{diagnostic}")
            consumed_sets[diagnostic].update(run_ids)
    union = (
        set(historical["excluded_state_identities"])
        | set.union(*suite_sets)
        | consumed_sets["396"]
        | consumed_sets["397"]
    )
    if len(union) != 94_370:
        raise ValueError(f"known_exclusion_union_unexpected:{len(union)}")
    union_path = DESTINATION / "exclusion-proof/known-excluded-state-identities.json"
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union_path.write_text(json.dumps(sorted(union), separators=(",", ":")) + "\n")
    rows.append(
        {
            "source_path": "recomputed:#396 historical union + actual/declared #396/#397 suites",
            "bundle_path": "exclusion-proof/known-excluded-state-identities.json",
            "sha256": sha256(union_path),
            "bytes": str(union_path.stat().st_size),
        }
    )
    verify_inventory(DESTINATION, rows)
    inventory = {"schema": "seed398-frozen-input-inventory-v1", "files": rows}
    (DESTINATION / "sha256-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "bundle": str(DESTINATION.relative_to(ROOT)),
        "inventory_count": len(rows),
        "historical_excluded_states": len(historical["excluded_state_identities"]),
        "suite_unique_state_counts": [len(states) for states in suite_sets],
        "suite_intersection": len(suite_sets[0] & suite_sets[1]),
        "consumed_identity_counts": {
            key: len(value) for key, value in consumed_sets.items()
        },
        "known_excluded_states": len(union),
        "runtime_contract": runtime_contract,
        "seed455_runtime_contract": seed455_runtime_contract,
    }
    provenance = {
        "schema": "seed398-frozen-input-provenance-v1",
        "source_checkout": str(ROOT),
        "historical_recovery_receipt": {
            "path": "docs/data/seed461-o0-checkpoint-recovery-provenance.json",
            "sha256": sha256(receipt_path),
            "method": receipt["method"],
            "original_execution_checkout": receipt["original_execution_checkout"],
        },
        "historical_registration_sha256": sha256(DATA / "registration.json"),
        "historical_binding_sha256": sha256(DATA / "evaluation-binding.json"),
        "seed455_checkpoint_sha256": sha256(seed455),
        "original_o0_e4_checkpoint_sha256": sha256(e4),
        "copied_inventory_sha256": sha256(DESTINATION / "sha256-inventory.json"),
        "bundle_identity": result,
    }
    (DESTINATION / "provenance-receipt.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    readiness = {
        "schema": "seed398-frozen-input-readiness-v1",
        "status": "ready_for_diagnostic_registration_after_source_freeze",
        "games_run": 0,
        "registration_created": False,
        "training_performed": False,
        "verified_inputs": result,
        "remaining_gaps": [],
        "scope_note": "This bundle restores historical inputs and exclusion evidence only; experiment execution sources and registration must still be finalized and hashed before registration.",
    }
    (DESTINATION / "readiness-report.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n"
    )
    (DESTINATION / "RESTORE.md").write_text(
        "# Frozen seed398 input bundle\n\n"
        "The bundle is self-contained and hash-inventoried. Historical source paths and hashes are recorded in `sha256-inventory.json`; relocated bundle paths are separate and do not rewrite historical registrations.\n\n"
        "Restore by copying the contents of this directory into the intended repository-local `.tmp/seed398-frozen-input-bundle/` directory, then run `python -m ml.alphazero_lite.prepare_seed398_recovery_bundle` from the checkout to verify original source identities and regenerate the readiness report. Use the archived `artifacts/original-o0-e4/` and `artifacts/seed455-opponent/` directories as evaluator inputs; use `artifacts/seed455-checkpoint.npz` only as checkpoint provenance. Native runtime files are under `runtime/`.\n\n"
        "To verify a copied bundle without access to original paths, load `sha256-inventory.json` and call `verify_inventory(bundle_root, inventory['files'])` from `ml.alphazero_lite.prepare_seed398_recovery_bundle`. The script does not register a suite or launch games.\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build_bundle(), indent=2, sort_keys=True))
