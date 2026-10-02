"""Reproduce the identity-level audit for the invalid seed397 evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.build_opening_suite import canonical_key, load_suite_jsonl
from ml.alphazero_lite.opening_exclusion_contract import (
    _source_identities,
    historical_opening_identities,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
OLD_SUITE = DATA / "seed397-o0-e1-e4-openings.jsonl"
VALID_SUITE = DATA / "seed461-e1-e4-corrected-diagnostic/seed397-openings-v2.jsonl"
MANIFEST = DATA / "order38615-a5-frozen-diagnostic-v4/opening-exclusion-manifest.json"
SUITES_395_396 = (
    DATA / "order38615-a5-frozen-diagnostic-v4/seed395-openings-v2.jsonl",
    DATA / "order38615-a5-frozen-diagnostic-v4/seed396-openings-v2.jsonl",
)
ERRATUM = DATA / "seed397-evidence-erratum-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_states(path: Path) -> set[str]:
    return {canonical_key(row["state"]) for row in load_suite_jsonl(str(path))}


def build_audit() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old_rows = load_suite_jsonl(str(OLD_SUITE))
    old_states = [canonical_key(row["state"]) for row in old_rows]
    excluded_legacy = set(manifest["excluded_state_identities"])
    collisions = [
        {"opening_index": i, "state_identity": identity}
        for i, identity in enumerate(old_states)
        if identity in set(manifest["declared_state_identities"])
    ]

    legacy_collisions: dict[str, list[str]] = {}
    source_identities: dict[str, set[str]] = {}
    for source in manifest["sources"]:
        if source["kind"] != "historical_suite":
            continue
        declared, actual = _source_identities(source)
        identities = set(declared) | set(actual)
        source_identities[source["path"]] = identities
        overlaps = sorted(set(old_states) & identities)
        if overlaps:
            legacy_collisions[source["path"]] = overlaps
    for collision in collisions:
        collision["historical_sources"] = sorted(
            path
            for path, identities in source_identities.items()
            if collision["state_identity"] in identities
        )

    complete_exclusion = set(excluded_legacy)
    for path in SUITES_395_396:
        declared, actual = historical_opening_identities(load_suite_jsonl(str(path)))
        complete_exclusion.update(declared)
        complete_exclusion.update(actual)
    old_exclusion = set(manifest["actual_state_identities"])
    for path in SUITES_395_396:
        _, actual = historical_opening_identities(load_suite_jsonl(str(path)))
        old_exclusion.update(actual)

    valid_ids = suite_states(VALID_SUITE)
    old_ids = set(old_states)
    report = {
        "schema": "seed397-evidence-erratum-audit-v1",
        "classification": "protocol_invalid",
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in (
                MANIFEST,
                OLD_SUITE,
                VALID_SUITE,
                *SUITES_395_396,
                DATA / "seed397-o0-e1-e4-registration.json",
                DATA / "seed397-o0-e1-e4-evaluation-binding.json",
            )
        },
        "excluded_state_counts": {
            "complete_registered_union": len(complete_exclusion),
            "seed397_union": len(old_exclusion),
            "omitted_count": len(complete_exclusion - old_exclusion),
            "omitted_state_identities": sorted(complete_exclusion - old_exclusion),
        },
        "historical_declared_collisions": {
            "count": len(collisions),
            "opening_indices_zero_based": [row["opening_index"] for row in collisions],
            "rows": collisions,
        },
        "shared_between_valid_396_and_invalid_397": {
            "count": len(valid_ids & old_ids),
            "state_identities": sorted(valid_ids & old_ids),
        },
        "historical_source_collision_counts": {
            path: len(identities) for path, identities in legacy_collisions.items()
        },
    }
    return report


def main() -> None:
    payload = json.dumps(build_audit(), indent=2, sort_keys=True) + "\n"
    if ERRATUM.exists() and ERRATUM.read_text(encoding="utf-8") != payload:
        raise ValueError("audit_output_conflict")
    ERRATUM.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
