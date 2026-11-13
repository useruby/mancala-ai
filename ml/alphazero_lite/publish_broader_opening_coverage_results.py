"""Publish reproducible results and an evidence digest for the frozen pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import broader_opening_coverage_pilot as pilot


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def evidence_manifest(data: Path) -> dict[str, Any]:
    named_paths = [
        (data / name, f"docs/data/broader-opening-coverage-pilot/{name}")
        for name in (
            "freeze.json",
            "game-assignments.json",
            "control-openings.json",
            "treatment-openings.json",
            "preflight.json",
        )
    ]
    named_paths += [
        (path, path.relative_to(pilot.ROOT).as_posix())
        for path in (
            pilot.INVENTORY_PATH,
            pilot.INVENTORY_SOURCE_PATH,
            *(pilot.ROOT / source for source in pilot.SOURCE_PATHS),
        )
    ]
    marker = data / "pairs/run-identity.v2"
    named_paths.append((marker, marker.relative_to(pilot.ROOT).as_posix()))
    named_paths += [
        (path, path.relative_to(pilot.ROOT).as_posix())
        for path in sorted((data / "pairs").glob("*.json"))
    ]
    named_paths += [
        (path, path.relative_to(pilot.ROOT).as_posix())
        for path in sorted((data / "attempts").glob("*.json"))
    ]
    files = []
    for path, relative in named_paths:
        content = path.read_bytes()
        files.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema": "broader_opening_pilot_raw_evidence_digest_v1",
        "files": files,
    }
    manifest["digest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest


def make_result(data: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = pilot.load_freeze(
        data / "freeze.json",
        "3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5",
    )
    analysis = pilot.analyze(freeze, data / "pairs", data / "attempts")
    jobs = pilot.jobs_from_freeze(freeze)
    committed, pending = pilot.load_pilot_pair_checkpoints(data / "pairs", jobs)
    if pending:
        raise ValueError("pilot_checkpoint_set_incomplete")

    supplied = {
        "canonical": {
            row["canonical_identity"]
            for row in freeze["control_manifest"] + freeze["treatment_manifest"]
        },
        "kalah_v3_float32": {
            row["production_float32_identity"]
            for row in freeze["control_manifest"] + freeze["treatment_manifest"]
        },
    }
    coverage: dict[str, Any] = {}
    for scheme in ("canonical", "kalah_v3_float32"):
        by_arm: dict[str, set[str]] = {}
        for arm in ("control", "treatment"):
            rows = [
                row
                for pair in committed
                if pair["arm"] == arm
                for game in pair["games"]
                for row in game["rows"]
            ]
            by_arm[arm] = pilot.coverage_identity_set(rows, scheme, supplied[scheme])
        shared = by_arm["control"] & by_arm["treatment"]
        coverage[scheme] = {
            "shared_distinct_positions": len(shared),
            "control_only_distinct_positions": len(by_arm["control"] - shared),
            "treatment_only_distinct_positions": len(by_arm["treatment"] - shared),
        }

    evidence = evidence_manifest(data)
    ratios = analysis["treatment_control_yield_ratio"]
    if any(value is None for value in ratios.values()):
        decision = "inconclusive_zero_control_yield"
    elif analysis["larger_run_gate_met"]:
        decision = "propose_larger_run"
    else:
        decision = "do_not_propose_larger_run"
    result = {
        "schema": "broader_opening_coverage_pilot_results_v1",
        "freeze_sha256": analysis["freeze_sha256"],
        "scope": analysis["scope"],
        "evidence_digest_sha256": evidence["digest_sha256"],
        "analysis": analysis,
        "coverage_overlap": coverage,
        "decision": decision,
        "conclusion": (
            "The frozen 62-opening pilot did not meet the 1.15x treatment/control "
            "yield gate under both identity schemes; do not propose a larger run. "
            "This conclusion applies only to the 62 matchable openings."
        ),
        "historical_novelty_scope": "authenticated published #556-#560 inventory only",
        "program_status": {
            "seed_580_decision_changed": False,
            "seed_483_reopened": False,
            "training_or_strength_screening_performed": False,
            "serving_changes_performed": False,
            "sealed_suite_accessed": False,
        },
    }
    return result, evidence


def main() -> None:
    data = pilot.DATA
    result, evidence = make_result(data)
    (data / "evidence-digest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (data / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
