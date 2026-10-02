"""Archive execution sources and verify post-execution A5 publication."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed461_arena_validation as validation
from ml.alphazero_lite.build_opening_suite import (
    load_suite_jsonl,
    validate_arena_entries,
)
from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity
from ml.alphazero_lite.opening_exclusion_contract import (
    create_manifest,
    validate_suite_set,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/order38615-a5-frozen-diagnostic-v4"
REG = DATA / "registration.json"
BIND = DATA / "evaluation-binding.json"
MANIFEST = DATA / "opening-exclusion-manifest.json"
SOURCE_BIND = ROOT / "docs/data/order38615-a5-confirmation-candidate-binding.json"
SNAPSHOT_MAP = DATA / "execution-source-snapshot-map.json"
AMENDMENT = DATA / "post-execution-publication-amendment.json"
SOURCE_PATHS = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/opening_exclusion_contract.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def archive_sources() -> dict[str, Any]:
    """Byte-copy registered execution sources before delivery formatting."""
    manifest = load_json(MANIFEST)
    manifest_sources = {row["path"]: row for row in manifest["sources"]}
    snapshot_dir = DATA / "execution-source-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mappings = []
    for source_name in SOURCE_PATHS:
        source = ROOT / source_name
        expected = manifest_sources[source_name]["sha256"]
        if sha(source) != expected:
            raise ValueError(f"registered_source_changed_before_snapshot:{source_name}")
        destination = snapshot_dir / f"{Path(source_name).name}.execution-source"
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError(f"execution_snapshot_conflict:{destination.name}")
        else:
            shutil.copyfile(source, destination)
        if sha(destination) != expected:
            raise ValueError(f"execution_snapshot_hash_mismatch:{source_name}")
        mappings.append(
            {
                "original_path": source_name,
                "snapshot_path": str(destination.relative_to(ROOT)),
                "registered_sha256": expected,
            }
        )
    result = {
        "schema": "order38615-a5-execution-source-snapshot-map-v1",
        "registration_sha256": sha(REG),
        "manifest_sha256": sha(MANIFEST),
        "sources": mappings,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if SNAPSHOT_MAP.exists() and SNAPSHOT_MAP.read_text() != payload:
        prior = load_json(SNAPSHOT_MAP)
        if prior["registration_sha256"] != result["registration_sha256"]:
            raise ValueError("execution_snapshot_map_conflict")
        for row in prior["sources"]:
            if sha(ROOT / row["snapshot_path"]) != row["registered_sha256"]:
                raise ValueError("prior_execution_snapshot_corrupted")
    SNAPSHOT_MAP.write_text(payload)
    return result


def _ast_dump(path: Path) -> str:
    tree = ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path), type_comments=True
    )
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def create_publication_amendment() -> dict[str, Any]:
    """Record post-execution hashes and AST-only formatting equivalence."""
    snapshot_map = load_json(SNAPSHOT_MAP)
    source_records = []
    for row in snapshot_map["sources"]:
        original = ROOT / row["original_path"]
        snapshot = ROOT / row["snapshot_path"]
        source_records.append(
            {
                **row,
                "formatted_sha256": sha(original),
                "ast_equivalent": _ast_dump(snapshot) == _ast_dump(original),
                "ast_method": "ast.parse(type_comments=True); ast.dump(annotate_fields=True, include_attributes=False)",
            }
        )
    if not all(row["ast_equivalent"] for row in source_records):
        raise ValueError("execution_source_ast_equivalence_failed")

    registration = load_json(REG)
    binding = load_json(BIND)
    results = load_json(DATA / "results.json")
    evidence = {
        "registration_sha256": sha(REG),
        "manifest_sha256": sha(MANIFEST),
        "evaluation_binding_sha256": sha(BIND),
        "candidate_binding_sha256": sha(SOURCE_BIND),
        "suite_sha256": {
            seed: spec["sha256"]
            for seed, spec in registration["evaluation"]["suites"].items()
        },
        "raw_reports_and_games": {
            seed: {
                "report_sha256": record["report_sha256"],
                "games_sha256": record["games_sha256"],
            }
            for seed, record in binding["reports"].items()
        },
        "raw_evidence_paths": {
            seed: {"report_path": record["report"], "games_path": record["games"]}
            for seed, record in binding["reports"].items()
        },
        "opening_score_matrix_sha256": sha(DATA / "opening-score-matrix.json"),
        "game_outcome_accounting_sha256": sha(DATA / "game-outcome-accounting.jsonl"),
        "results_sha256": sha(DATA / "results.json"),
        "canonical_gate_result_sha256": sha(
            ROOT / "docs/data/order38615-a5-promotion-review-result.json"
        ),
        "canonical_gate_score_matrix_sha256": sha(
            ROOT / "docs/data/order38615-a5-canonical-gate/opening-score-matrix.json"
        ),
        "public_results_binding": {
            "registration_sha256": results["registration_sha256"],
            "exclusion_manifest_sha256": results["exclusion_manifest_sha256"],
            "evaluation_binding_sha256": results["evaluation_binding_sha256"],
        },
    }
    amendment = {
        "schema": "order38615-a5-post-execution-publication-amendment-v1",
        "status": "post_execution_formatting_only",
        "original_hashes": evidence,
        "execution_sources": source_records,
        "ast_equivalence": {
            "method": "Parse archived and formatted Python with type_comments=True; compare ast.dump(..., annotate_fields=True, include_attributes=False).",
            "source_locations_excluded": True,
            "type_comments_retained": True,
            "all_sources_equivalent": True,
        },
        "separate_functional_publication_verifier": {
            "path": "ml/alphazero_lite/publish_order38615_a5_diagnostic.py",
            "sha256": sha(Path(__file__).resolve()),
            "role": "post-execution publication verification only; launch validation remains bound to exact original source bytes",
        },
        "statement": "Source formatting occurred after the 2,048 games and completed analysis. Each delivery source has the same Python AST as its byte-exact execution snapshot; no experimental input, runtime, suite, game, outcome, registration, manifest, or preregistered hash was changed.",
    }
    payload = json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    if AMENDMENT.exists() and AMENDMENT.read_text() != payload:
        prior = load_json(AMENDMENT)
        if (
            any(
                amendment["original_hashes"].get(key) != value
                for key, value in prior.get("original_hashes", {}).items()
            )
            or prior.get("execution_sources") != amendment["execution_sources"]
        ):
            raise ValueError("publication_amendment_conflict")
    AMENDMENT.write_text(payload)
    return amendment


def validate_publication() -> dict[str, Any]:
    """Verify frozen data against snapshots, without weakening launch checks."""
    registration = load_json(REG)
    binding = load_json(BIND)
    manifest = load_json(MANIFEST)
    snapshot_map = load_json(SNAPSHOT_MAP)
    amendment = load_json(AMENDMENT)
    verifier_record = amendment["separate_functional_publication_verifier"]
    if sha(ROOT / verifier_record["path"]) != verifier_record["sha256"]:
        raise ValueError("publication_verifier_hash_mismatch")
    if sha(REG) != amendment["original_hashes"]["registration_sha256"]:
        raise ValueError("publication_registration_hash_mismatch")
    if sha(MANIFEST) != amendment["original_hashes"]["manifest_sha256"]:
        raise ValueError("publication_manifest_hash_mismatch")
    if sha(BIND) != amendment["original_hashes"]["evaluation_binding_sha256"]:
        raise ValueError("publication_binding_hash_mismatch")
    if snapshot_map["registration_sha256"] != sha(REG):
        raise ValueError("snapshot_registration_binding_mismatch")
    if snapshot_map["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("snapshot_manifest_binding_mismatch")

    snapshots: dict[str, dict[str, Any]] = {}
    amendment_sources = {
        row["original_path"]: row for row in amendment["execution_sources"]
    }
    for row in snapshot_map["sources"]:
        original = row["original_path"]
        snapshot = ROOT / row["snapshot_path"]
        formatted = ROOT / original
        if sha(snapshot) != row["registered_sha256"]:
            raise ValueError(f"execution_snapshot_hash_mismatch:{original}")
        publication_record = amendment_sources[original]
        if sha(formatted) != publication_record["formatted_sha256"]:
            raise ValueError(f"formatted_delivery_hash_mismatch:{original}")
        snapshot_ast = _ast_dump(snapshot)
        delivery_ast = _ast_dump(formatted)
        if snapshot_ast != delivery_ast:
            raise ValueError(f"execution_source_ast_mismatch:{original}")
        if not publication_record["ast_equivalent"]:
            raise ValueError(f"amendment_ast_assertion_false:{original}")
        snapshots[original] = row

    recomputed = create_manifest(
        [{"path": row["path"], "kind": row["kind"]} for row in manifest["sources"]]
    )
    expected_code_hashes = {
        row["original_path"]: row["registered_sha256"]
        for row in snapshot_map["sources"]
    }
    for record in recomputed["sources"]:
        if record["path"] in expected_code_hashes:
            record["sha256"] = expected_code_hashes[record["path"]]
    if recomputed != manifest:
        raise ValueError("publication_exclusion_sources_or_states_mismatch")

    all_suites = {
        seed: load_suite_jsonl(str(ROOT / spec["path"]))
        for seed, spec in registration["evaluation"]["suites"].items()
    }
    for seed, entries in all_suites.items():
        spec = registration["evaluation"]["suites"][seed]
        if sha(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"publication_suite_hash_mismatch:{seed}")
        if len(validate_arena_entries(entries)) != spec["opening_count"]:
            raise ValueError(f"publication_suite_replay_mismatch:{seed}")
    suite_identities = validate_suite_set(all_suites, manifest)

    candidate = registration["candidate"]
    opponent = registration["opponent"]
    source_binding_path = SOURCE_BIND
    if (
        sha(source_binding_path)
        != amendment["original_hashes"]["candidate_binding_sha256"]
    ):
        raise ValueError("publication_candidate_binding_hash_mismatch")
    source_binding = load_json(source_binding_path)
    if (
        source_binding["candidate"]["checkpoint_sha256"]
        != candidate["checkpoint_sha256"]
    ):
        raise ValueError("publication_checkpoint_identity_mismatch")
    for name, digest in candidate["artifact_sha256"].items():
        if sha(Path(candidate["artifact"]) / name) != digest:
            raise ValueError(f"publication_candidate_artifact_hash_mismatch:{name}")
    validate_frozen_opponent_identity(
        Path(opponent["artifact"]), opponent, candidate["runtime_contract"]
    )

    for seed, spec in registration["evaluation"]["suites"].items():
        record = binding["reports"][seed]
        report_path = Path(record["report"])
        games_path = Path(record["games"])
        if (
            sha(report_path) != record["report_sha256"]
            or sha(games_path) != record["games_sha256"]
        ):
            raise ValueError(f"publication_raw_evidence_hash_mismatch:{seed}")
        report = load_json(report_path)
        game_rows = [
            json.loads(line)
            for line in games_path.read_text().splitlines()
            if line.strip()
        ]
        validation.validate_arena_evidence(
            report,
            game_rows,
            all_suites[seed],
            seed,
            {
                "artifact": candidate["artifact"],
                "runtime_contract": candidate["runtime_contract"],
            },
            opponent,
            {
                **registration["evaluation"],
                "games_per_candidate": registration["evaluation"]["games_per_suite"],
                "suite": spec,
                "arena_seed": int(seed),
                "seed_contract": registration["evaluation"]["seed_contract"],
            },
        )
        if len(suite_identities[seed]) != 512 or len(game_rows) != 1024:
            raise ValueError(f"publication_suite_or_game_accounting_mismatch:{seed}")

    public_results = load_json(DATA / "results.json")
    matrix_path = DATA / "opening-score-matrix.json"
    accounting_path = DATA / "game-outcome-accounting.jsonl"
    if sha(matrix_path) != public_results["opening_score_matrix_sha256"]:
        raise ValueError("publication_score_matrix_hash_mismatch")
    if sha(accounting_path) != public_results["game_outcome_accounting_sha256"]:
        raise ValueError("publication_game_accounting_hash_mismatch")
    if public_results["total_games"] != 2048:
        raise ValueError("publication_game_total_mismatch")
    return {
        "status": "verified_post_execution_publication",
        "registration_sha256": sha(REG),
        "manifest_sha256": sha(MANIFEST),
        "evaluation_binding_sha256": sha(BIND),
        "snapshot_sources_verified": len(snapshots),
        "excluded_state_count": manifest["excluded_state_count"],
        "total_games": public_results["total_games"],
        "diagnostic_pass": public_results["per_suite_and_pooled_pass"],
        "launch_validation_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("archive", "amend", "verify"))
    args = parser.parse_args()
    if args.action == "archive":
        result = archive_sources()
    elif args.action == "amend":
        result = create_publication_amendment()
    else:
        result = validate_publication()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
