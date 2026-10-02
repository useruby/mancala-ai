"""Frozen four-treatment seed398 policy/value composition arena.

This diagnostic composes complete evaluator outputs through their own trunks.
It does not edit weights, train checkpoints, or export hybrid artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_arena_validation as evidence_validation
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.arena import apply_opening_moves, canonical_game_state_hash
from ml.alphazero_lite.build_opening_suite import INITIAL_STATE, canonical_key
from ml.alphazero_lite.frozen_opponent_identity import (
    validate_frozen_opponent_identity,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.opening_exclusion_contract import (
    create_manifest,
    historical_opening_identities,
    validate_suite_against_manifest,
    write_immutable_json,
)
from ml.alphazero_lite.prepare_seed398_recovery_bundle import (
    EXPECTED_E4,
    EXPECTED_SEED455,
    sha256,
    verify_inventory,
)
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-policy-value-composition"
WORK = ROOT / ".tmp/seed398-policy-value-composition"
BUNDLE = ROOT / ".tmp/seed398-frozen-input-bundle"
HISTORICAL_DATA = ROOT / "docs/data/seed461-e1-e4-corrected-diagnostic"
E4_CHECKPOINT = ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz"
E4_ARTIFACT = ROOT / ".tmp/seed461-order-confirmation/artifacts/O0-E4"
SEED455_CHECKPOINT = (
    ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz"
)
SEED455_ARTIFACT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
HISTORICAL_MANIFEST = HISTORICAL_DATA / "opening-exclusion-manifest.json"
HISTORICAL_SOURCES = (
    "ml/alphazero_lite/seed398_composition_diagnostic.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/seed461_arena_validation.py",
    "ml/alphazero_lite/opening_exclusion_contract.py",
    "ml/alphazero_lite/seed461_order_population.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/frozen_opponent_identity.py",
    "ml/alphazero_lite/prepare_seed398_recovery_bundle.py",
)
EXECUTION_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
).stdout.strip()

TREATMENTS = {
    "SS": {"policy_source": "seed455", "value_source": "seed455"},
    "FS": {"policy_source": "original_o0_e4", "value_source": "seed455"},
    "SF": {"policy_source": "seed455", "value_source": "original_o0_e4"},
    "FF": {"policy_source": "original_o0_e4", "value_source": "original_o0_e4"},
}
SOURCE_ARTIFACTS = {"seed455": SEED455_ARTIFACT, "original_o0_e4": E4_ARTIFACT}
PRIOR_SUITES = (
    HISTORICAL_DATA / "seed397-openings-v2.jsonl",
    ROOT / "docs/data/seed397-o0-e1-e4-openings.jsonl",
)
CONSUMED_LEDGERS = {
    "396_E1": ROOT / ".tmp/seed461-e1-e4-corrected-diagnostic/E1-games.jsonl",
    "396_E4": ROOT / ".tmp/seed461-e1-e4-corrected-diagnostic/E4-games.jsonl",
    "397_E1": ROOT / ".tmp/seed397-o0-e1-e4/E1-games.jsonl",
    "397_E4": ROOT / ".tmp/seed397-o0-e1-e4/E4-games.jsonl",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"artifact_identity_mismatch:{label}")


def cached_evidence_mode(
    report_exists: bool, games_exists: bool, record: dict[str, Any] | None
) -> str:
    """Classify cached evidence without ever treating partial data as recoverable."""
    any_file = report_exists or games_exists
    if not any_file:
        if record is not None:
            raise ValueError("bound_evidence_missing")
        return "launch"
    if not (report_exists and games_exists) or record is None:
        raise ValueError("partial_or_unbound_evidence")
    if record.get("state") == "running":
        return "recover"
    return "validate"


def validate_command_source_selection(
    command: list[str], treatment: dict[str, Any]
) -> None:
    expected = {
        "--challenger-policy-artifact": treatment["policy_artifact"],
        "--challenger-value-artifact": treatment["value_artifact"],
        "--current-policy-artifact": str(SEED455_ARTIFACT),
        "--current-value-artifact": str(SEED455_ARTIFACT),
    }
    for flag, path in expected.items():
        try:
            actual = command[command.index(flag) + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"command_component_source_missing:{flag}") from error
        if actual != path:
            raise ValueError(f"command_component_source_mismatch:{flag}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def execution_source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in HISTORICAL_SOURCES}


def verify_bundle() -> None:
    inventory = read_json(BUNDLE / "sha256-inventory.json")
    verify_inventory(BUNDLE, inventory["files"])
    readiness = read_json(BUNDLE / "readiness-report.json")
    if (
        readiness.get("status")
        != "ready_for_diagnostic_registration_after_source_freeze"
    ):
        raise ValueError("recovery_bundle_not_ready")


def component_bindings() -> dict[str, Any]:
    """Check and return exact immutable component/runtime source identities."""
    verify_bundle()
    e4_checkpoint_hash = sha256(E4_CHECKPOINT)
    require_hash(e4_checkpoint_hash, EXPECTED_E4, "original_o0_e4_checkpoint")
    require_hash(sha256(E4_ARTIFACT / "model.npz"), EXPECTED_E4, "original_o0_e4_model")
    e4_hashes = {
        name: sha256(E4_ARTIFACT / name)
        for name in ("model.npz", "weights.json", "metadata.json", "search_policy.json")
    }
    seed455_checkpoint_hash = sha256(SEED455_CHECKPOINT)
    require_hash(seed455_checkpoint_hash, EXPECTED_SEED455, "seed455_checkpoint")
    seed455_hashes = {
        name: sha256(SEED455_ARTIFACT / name)
        for name in ("weights.json", "metadata.json", "search_policy.json")
    }
    metadata = read_json(SEED455_ARTIFACT / "metadata.json")
    if metadata["artifacts"]["weights_sha256"] != EXPECTED_SEED455:
        raise ValueError("seed455_runtime_checkpoint_lineage_mismatch")
    opponent = {
        "artifact": str(SEED455_ARTIFACT),
        "weights_sha256": seed455_hashes["weights.json"],
        "metadata_sha256": seed455_hashes["metadata.json"],
        "sidecar_sha256": seed455_hashes["search_policy.json"],
        "native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        "tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
    }
    runtime_contract = resolve_strength_comparison_runtime_contract(
        current_artifact=SEED455_ARTIFACT,
        challenger_artifact=E4_ARTIFACT,
    )
    self_contract = resolve_strength_comparison_runtime_contract(
        current_artifact=SEED455_ARTIFACT,
        challenger_artifact=SEED455_ARTIFACT,
    )
    if runtime_contract != self_contract:
        raise ValueError("component_runtime_contract_mismatch")
    validate_frozen_opponent_identity(SEED455_ARTIFACT, opponent, runtime_contract)
    for filename, expected in e4_hashes.items():
        if sha256(E4_ARTIFACT / filename) != expected:
            raise ValueError(f"e4_artifact_changed:{filename}")
    return {
        "seed455": {
            "checkpoint": str(SEED455_CHECKPOINT),
            "checkpoint_sha256": seed455_checkpoint_hash,
            "artifact": str(SEED455_ARTIFACT),
            "artifact_sha256": seed455_hashes,
            "architecture": metadata["architecture"],
        },
        "original_o0_e4": {
            "checkpoint": str(E4_CHECKPOINT),
            "checkpoint_sha256": e4_checkpoint_hash,
            "artifact": str(E4_ARTIFACT),
            "artifact_sha256": e4_hashes,
            "architecture": read_json(E4_ARTIFACT / "metadata.json")["architecture"],
        },
        "native_runtime_contract": runtime_contract,
    }


def _read_game_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def actual_consumed_identities(path: Path) -> tuple[set[str], set[str]]:
    """Recover raw arena-consumed canonical starts without suite substitution."""
    declared_hashes: set[str] = set()
    actual_states: set[str] = set()
    rows = _read_game_rows(path)
    for index, row in enumerate(rows):
        prefix = [int(move) for move in row["opening_prefix_moves"]]
        applied_count = row.get("opening_applied_prefix_length")
        if not isinstance(applied_count, int) or isinstance(applied_count, bool):
            raise ValueError(f"consumed_prefix_length_missing:{path.name}:{index}")
        if not 0 <= applied_count <= len(prefix):
            raise ValueError(f"consumed_prefix_length_invalid:{path.name}:{index}")
        game = KalahGame.from_state(INITIAL_STATE)
        applied = apply_opening_moves(game, prefix[:applied_count])
        if applied != applied_count or canonical_game_state_hash(game) != row.get(
            "opening_state_hash"
        ):
            raise ValueError(f"consumed_state_hash_mismatch:{path.name}:{index}")
        if row.get("opening_contract") != "arena_player_relative_v2":
            raise ValueError(f"consumed_opening_contract_mismatch:{path.name}:{index}")
        actual_states.add(canonical_key(game.to_state()))
        declared_hashes.add(str(row["opening_state_hash"]))
    return actual_states, declared_hashes


def build_complete_exclusion_proof() -> dict[str, Any]:
    """Recompute #396's full union plus both suites and observed consumptions."""
    historical = read_json(HISTORICAL_MANIFEST)
    base = create_manifest(
        [{"path": row["path"], "kind": row["kind"]} for row in historical["sources"]]
    )
    if base != historical or len(base["excluded_state_identities"]) != 93_366:
        raise ValueError("historical_396_exclusion_proof_invalid")
    declared_union = set(base["declared_state_identities"])
    actual_union = set(base["actual_state_identities"])
    add_sources: list[dict[str, Any]] = []
    for suite_path in PRIOR_SUITES:
        if not suite_path.is_file():
            raise FileNotFoundError(f"completed_suite_missing:{suite_path}")
        declared, actual = historical_opening_identities(
            suites.load_suite_jsonl(str(suite_path))
        )
        declared_union.update(declared)
        actual_union.update(actual)
        add_sources.append(
            {
                "kind": "completed_evaluation_suite",
                "path": str(suite_path.relative_to(ROOT)),
                "sha256": sha256(suite_path),
                "declared_identities": sorted(declared),
                "actual_identities": sorted(actual),
            }
        )
    consumed_evidence: dict[str, Any] = {}
    for label, path in CONSUMED_LEDGERS.items():
        if not path.is_file():
            raise FileNotFoundError(f"completed_consumed_ledger_missing:{label}")
        states, observed_hashes = actual_consumed_identities(path)
        actual_union.update(states)
        consumed_evidence[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "game_rows": len(_read_game_rows(path)),
            "unique_consumed_states": len(states),
            "observed_opening_state_hashes": sorted(observed_hashes),
            "consumed_state_identities": sorted(states),
        }
    excluded = declared_union | actual_union
    if len(excluded) < 94_370:
        raise ValueError(f"known_exclusion_union_incomplete:{len(excluded)}")
    return {
        "schema": "seed398-complete-opening-exclusion-proof-v1",
        "historical_396_manifest": {
            "path": str(HISTORICAL_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(HISTORICAL_MANIFEST),
            "excluded_state_count": len(base["excluded_state_identities"]),
            "excluded_identity_sha256": base["excluded_identity_sha256"],
        },
        "additional_completed_suites": add_sources,
        "consumed_evidence": consumed_evidence,
        "declared_state_identities": sorted(declared_union),
        "actual_state_identities": sorted(actual_union),
        "excluded_state_identities": sorted(excluded),
        "excluded_state_count": len(excluded),
        "excluded_identity_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode()
        ).hexdigest(),
        "known_base_count": 94_370,
        "additional_consumed_state_count_beyond_known_base": len(excluded) - 94_370,
    }


def source_snapshots(hashes: dict[str, str]) -> None:
    directory = DATA / "execution-source-snapshots"
    for name, digest in hashes.items():
        source = ROOT / name
        if sha256(source) != digest:
            raise ValueError(f"execution_source_changed_before_snapshot:{name}")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != digest:
            raise ValueError(f"immutable_execution_snapshot_conflict:{name}")
        if not target.exists():
            target.write_bytes(source.read_bytes())


def _strict_suite(rows: list[dict[str, Any]], excluded: set[str]) -> set[str]:
    if len(rows) != 512:
        raise ValueError("suite_opening_count_mismatch")
    actual_ids = suites.validate_arena_entries(rows)
    if len(set(actual_ids)) != 512:
        raise ValueError("suite_consumed_states_not_unique")
    validate_suite_against_manifest(
        rows,
        {
            "declared_state_identities": sorted(excluded),
            "actual_state_identities": sorted(excluded),
        },
    )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        game = KalahGame.from_state(INITIAL_STATE)
        prefix = [int(move) for move in row["prefix_moves"]]
        applied = apply_opening_moves(game, prefix)
        if applied != len(prefix):
            raise ValueError(f"suite_replay_truncated:{index}")
        identity = canonical_key(game.to_state())
        if identity != row["state_hash"] or identity in seen:
            raise ValueError(f"suite_declared_or_consumed_identity_invalid:{index}")
        if len(prefix) > 8 or game.over() or sum(game.pits) <= 32:
            raise ValueError(f"suite_opening_ineligible:{index}")
        seen.add(identity)
    if seen & excluded:
        raise ValueError("suite_exclusion_overlap")
    return seen


def treatment_bindings(components: dict[str, Any]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for name, treatment in TREATMENTS.items():
        policy = components[treatment["policy_source"]]
        value = components[treatment["value_source"]]
        bindings[name] = {
            **treatment,
            "policy_artifact": policy["artifact"],
            "policy_checkpoint_sha256": policy["checkpoint_sha256"],
            "policy_model_sha256": policy["artifact_sha256"].get(
                "model.npz", policy["checkpoint_sha256"]
            ),
            "policy_weights_sha256": policy["artifact_sha256"]["weights.json"],
            "value_artifact": value["artifact"],
            "value_checkpoint_sha256": value["checkpoint_sha256"],
            "value_model_sha256": value["artifact_sha256"].get(
                "model.npz", value["checkpoint_sha256"]
            ),
            "value_weights_sha256": value["artifact_sha256"]["weights.json"],
            "trunk_composition": "each output computed by its own complete source evaluator; no weight splicing",
        }
    return bindings


def register() -> None:
    components = component_bindings()
    proof = build_complete_exclusion_proof()
    code_hashes = execution_source_hashes()
    source_snapshots(code_hashes)
    excluded = set(proof["excluded_state_identities"])
    selected = population.select_holdout(excluded, seed=398, size=512)
    rows = [suites.export_arena_entry(row) for row in selected]
    _strict_suite(rows, excluded)
    suite_path = DATA / "seed398-openings-v2.jsonl"
    suite_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if suite_path.exists() and suite_path.read_text(encoding="utf-8") != suite_payload:
        raise ValueError("immutable_seed398_suite_conflict")
    if not suite_path.exists():
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.write_text(suite_payload, encoding="utf-8")
    proof_path = DATA / "opening-exclusion-proof.json"
    write_immutable_json(proof_path, proof)
    treatment_map = treatment_bindings(components)
    evaluation = {
        "opening_count": 512,
        "games_per_treatment": 1024,
        "games_per_opening": 2,
        "both_seats_per_opening": True,
        "games_total": 4096,
        "simulations_per_side": 384,
        "c_puct": 1.25,
        "base_seed": 398,
        "seed_contract": "azlite_eval_seed_v2",
        "workers": 24,
        "opening_contract": "arena_player_relative_v2",
        "outcome_dependent_extensions": False,
    }
    registration = {
        "schema": "seed398-policy-value-composition-registration-v1",
        "status": "registered_before_games",
        "execution_commit": EXECUTION_COMMIT,
        "execution_source_hashes": code_hashes,
        "component_sources": components,
        "treatments": treatment_map,
        "exclusion_proof": {
            "path": str(proof_path.relative_to(ROOT)),
            "sha256": sha256(proof_path),
            "excluded_state_count": proof["excluded_state_count"],
            "excluded_identity_sha256": proof["excluded_identity_sha256"],
        },
        "evaluation": {
            **evaluation,
            "suite": {
                "path": str(suite_path.relative_to(ROOT)),
                "sha256": sha256(suite_path),
                "opening_count": len(rows),
                "unique_states": 512,
                "selection_seed": 398,
            },
        },
        "analysis": {
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 398,
            "cluster": "opening; four treatment scores and both seats resampled together",
            "primary_interval_quantiles": [0.0125, 0.9875],
            "descriptive_interval_quantiles": [0.025, 0.975],
            "positive_followup_threshold": 0.03,
            "followup_rule": "primary mean >= +0.03 and 97.5% interval lower bound > 0",
            "harmful_classification": "mean <= -0.03 and interval upper bound < 0",
        },
    }
    write_immutable_json(DATA / "registration.json", registration)
    print(
        json.dumps(
            {
                "registration_sha256": sha256(DATA / "registration.json"),
                "suite_sha256": sha256(suite_path),
                "excluded_states": proof["excluded_state_count"],
                "openings": len(rows),
            },
            indent=2,
        )
    )


def validate_frozen(
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if execution_source_hashes() != registration["execution_source_hashes"]:
        raise ValueError("execution_source_hash_mismatch")
    components = component_bindings()
    if components != registration["component_sources"]:
        raise ValueError("component_source_identity_changed")
    proof_path = ROOT / registration["exclusion_proof"]["path"]
    proof = read_json(proof_path)
    recomputed = build_complete_exclusion_proof()
    if (
        proof != recomputed
        or sha256(proof_path) != registration["exclusion_proof"]["sha256"]
    ):
        raise ValueError("complete_exclusion_proof_changed")
    suite_path = ROOT / registration["evaluation"]["suite"]["path"]
    if sha256(suite_path) != registration["evaluation"]["suite"]["sha256"]:
        raise ValueError("frozen_suite_hash_mismatch")
    openings = suites.load_suite_jsonl(str(suite_path))
    _strict_suite(openings, set(proof["excluded_state_identities"]))
    if len(openings) != 512:
        raise ValueError("frozen_suite_count_mismatch")
    return components, proof, openings


def bind() -> None:
    registration_path = DATA / "registration.json"
    registration = read_json(registration_path)
    if registration.get("status") != "registered_before_games":
        raise ValueError("registration_status_invalid")
    _, proof, openings = validate_frozen(registration)
    bindings = {
        "schema": "seed398-policy-value-composition-binding-v1",
        "registration_sha256": sha256(registration_path),
        "suite_sha256": registration["evaluation"]["suite"]["sha256"],
        "exclusion_proof_sha256": sha256(DATA / "opening-exclusion-proof.json"),
        "execution_source_hashes": registration["execution_source_hashes"],
        "treatments": registration["treatments"],
        "reports": {},
        "status": "bound_before_games",
        "excluded_state_count": proof["excluded_state_count"],
        "opening_count": len(openings),
    }
    write_immutable_json(DATA / "evaluation-binding.json", bindings)
    print(f"binding_sha256={sha256(DATA / 'evaluation-binding.json')}")


def _expected_notes(
    treatment: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "challenger_path": treatment["policy_artifact"],
        "current_path": str(SEED455_ARTIFACT),
        "suite_sha256": evaluation["suite"]["sha256"],
        "seed": 398,
        "base_seed": 398,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
    }


def validate_treatment_evidence(
    name: str,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    registration: dict[str, Any],
    components: dict[str, Any],
) -> np.ndarray:
    treatment = registration["treatments"][name]
    evaluation = registration["evaluation"]
    notes = report.get("notes")
    if not isinstance(notes, dict):
        raise ValueError(f"report_notes_missing:{name}")
    for key, expected in _expected_notes(treatment, evaluation).items():
        if notes.get(key) != expected:
            raise ValueError(f"report_composition_identity_mismatch:{name}:{key}")
    candidate = {
        "artifact": treatment["policy_artifact"],
        "runtime_contract": components["native_runtime_contract"],
    }
    scores = evidence_validation.validate_arena_evidence(
        report,
        rows,
        openings,
        name,
        candidate,
        {"artifact": str(SEED455_ARTIFACT)},
        {
            **evaluation,
            "games_per_candidate": 1024,
            "suite": evaluation["suite"],
            "arena_seed": 398,
            "seed_contract": "azlite_eval_seed_v2",
        },
    )
    if len(scores) != 512:
        raise ValueError(f"treatment_opening_score_count_mismatch:{name}")
    return scores


def _treatment_paths(name: str) -> tuple[Path, Path]:
    return WORK / f"{name}.json", WORK / f"{name}-games.jsonl"


def _validate_existing_evidence(
    name: str,
    record: dict[str, Any],
    registration: dict[str, Any],
    components: dict[str, Any],
    openings: list[dict[str, Any]],
) -> np.ndarray:
    report_path, games_path = _treatment_paths(name)
    if not report_path.is_file() or not games_path.is_file():
        raise ValueError(f"partial_or_missing_evidence:{name}")
    if sha256(report_path) != record.get("report_sha256") or sha256(
        games_path
    ) != record.get("games_sha256"):
        raise ValueError(f"cached_evidence_hash_mismatch:{name}")
    return validate_treatment_evidence(
        name,
        read_json(report_path),
        _read_game_rows(games_path),
        openings,
        registration,
        components,
    )


def _command(
    name: str, registration: dict[str, Any], report_path: Path, games_path: Path
) -> list[str]:
    treatment = registration["treatments"][name]
    evaluation = registration["evaluation"]
    suite_path = ROOT / evaluation["suite"]["path"]
    return [
        sys.executable,
        str(ROOT / "ml/alphazero_lite/arena.py"),
        "--challenger",
        treatment["policy_artifact"],
        "--current",
        str(SEED455_ARTIFACT),
        "--challenger-policy-artifact",
        treatment["policy_artifact"],
        "--challenger-value-artifact",
        treatment["value_artifact"],
        "--current-policy-artifact",
        str(SEED455_ARTIFACT),
        "--current-value-artifact",
        str(SEED455_ARTIFACT),
        "--games",
        "1024",
        "--games-per-opening",
        "2",
        "--opening-prefixes-jsonl",
        str(suite_path),
        "--suite-sha256",
        evaluation["suite"]["sha256"],
        "--challenger-simulations",
        "384",
        "--current-simulations",
        "384",
        "--seed",
        "398",
        "--workers",
        "24",
        "--c-puct",
        "1.25",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--game-jsonl",
        str(games_path),
        "--out",
        str(report_path),
    ]


def run() -> None:
    registration_path = DATA / "registration.json"
    registration = read_json(registration_path)
    binding_path = DATA / "evaluation-binding.json"
    binding = read_json(binding_path)
    if binding.get("status") not in {
        "bound_before_games",
        "running",
        "completed_4096_games",
    }:
        raise ValueError("binding_status_invalid")
    components, proof, openings = validate_frozen(registration)
    if binding["registration_sha256"] != sha256(registration_path):
        raise ValueError("binding_registration_mismatch")
    if binding["exclusion_proof_sha256"] != sha256(
        DATA / "opening-exclusion-proof.json"
    ):
        raise ValueError("binding_exclusion_proof_mismatch")
    if binding["suite_sha256"] != registration["evaluation"]["suite"]["sha256"]:
        raise ValueError("binding_suite_mismatch")
    if binding["execution_source_hashes"] != execution_source_hashes():
        raise ValueError("binding_execution_sources_mismatch")
    excluded = set(proof["excluded_state_identities"])
    if excluded & {row["state_hash"] for row in openings}:
        raise ValueError("suite_exclusion_overlap_before_launch")
    binding["status"] = "running"
    for name in TREATMENTS:
        report_path, games_path = _treatment_paths(name)
        record = binding["reports"].get(name)
        evidence_mode = cached_evidence_mode(
            report_path.exists(), games_path.exists(), record
        )
        if evidence_mode == "recover":
            expected_command = _command(name, registration, report_path, games_path)
            if record.get("command") != expected_command:
                raise ValueError(f"recovery_command_identity_mismatch:{name}")
            validate_command_source_selection(
                expected_command, registration["treatments"][name]
            )
            # A process may have completed before the runner persisted hashes.
            scores = validate_treatment_evidence(
                name,
                read_json(report_path),
                _read_game_rows(games_path),
                openings,
                registration,
                components,
            )
            record.update(
                {
                    "state": "validated_recovered",
                    "report": str(report_path),
                    "report_sha256": sha256(report_path),
                    "games": str(games_path),
                    "games_sha256": sha256(games_path),
                    "score": float(scores.mean()),
                }
            )
            binding["reports"][name] = record
            write_json(binding_path, binding)
            continue
        if evidence_mode == "validate":
            expected_command = _command(name, registration, report_path, games_path)
            if record.get("command") != expected_command:
                raise ValueError(f"bound_command_identity_mismatch:{name}")
            validate_command_source_selection(
                expected_command, registration["treatments"][name]
            )
            _validate_existing_evidence(
                name, record, registration, components, openings
            )
            continue
        # Identity and exclusions are recomputed before every subprocess launch.
        components, proof, openings = validate_frozen(registration)
        command = _command(name, registration, report_path, games_path)
        treatment = registration["treatments"][name]
        validate_command_source_selection(command, treatment)
        binding["reports"][name] = {
            "state": "running",
            "command": command,
            "policy_source": treatment["policy_source"],
            "value_source": treatment["value_source"],
            "policy_checkpoint_sha256": treatment["policy_checkpoint_sha256"],
            "value_checkpoint_sha256": treatment["value_checkpoint_sha256"],
            "report": str(report_path),
            "games": str(games_path),
        }
        write_json(binding_path, binding)
        subprocess.run(command, cwd=ROOT, check=True)
        scores = validate_treatment_evidence(
            name,
            read_json(report_path),
            _read_game_rows(games_path),
            openings,
            registration,
            components,
        )
        record = binding["reports"][name]
        record.update(
            {
                "state": "completed_validated",
                "report_sha256": sha256(report_path),
                "games_sha256": sha256(games_path),
                "score": float(scores.mean()),
            }
        )
        write_json(binding_path, binding)
    if set(binding["reports"]) != set(TREATMENTS):
        raise ValueError("treatment_evidence_incomplete")
    for name, record in binding["reports"].items():
        _validate_existing_evidence(name, record, registration, components, openings)
    binding["status"] = "completed_4096_games"
    write_json(binding_path, binding)


def _outcome_rows(name: str, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in raw_rows:
        output.append({"treatment": name, **row})
    return output


def analyze() -> dict[str, Any]:
    registration = read_json(DATA / "registration.json")
    binding = read_json(DATA / "evaluation-binding.json")
    components, proof, openings = validate_frozen(registration)
    if binding.get("status") != "completed_4096_games":
        raise ValueError("diagnostic_not_complete")
    scores: dict[str, np.ndarray] = {}
    games_by_treatment: dict[str, list[dict[str, Any]]] = {}
    ledger: list[dict[str, Any]] = []
    for name in TREATMENTS:
        record = binding["reports"].get(name)
        if record is None:
            raise ValueError(f"treatment_binding_missing:{name}")
        report_path, games_path = _treatment_paths(name)
        scores[name] = _validate_existing_evidence(
            name, record, registration, components, openings
        )
        raw = _read_game_rows(games_path)
        games_by_treatment[name] = raw
        ledger.extend(_outcome_rows(name, raw))
    ledger.sort(key=lambda row: (row["treatment"], row["game_index"]))
    ledger_path = DATA / "validated-outcome-ledger.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger),
        encoding="utf-8",
    )
    matrix_rows = [
        {
            "opening_index": index,
            "opening_state_hash": openings[index]["state_hash"],
            **{name: float(scores[name][index]) for name in TREATMENTS},
        }
        for index in range(512)
    ]
    matrix = {
        "schema": "seed398-four-treatment-opening-score-matrix-v1",
        "registration_sha256": sha256(DATA / "registration.json"),
        "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
        "outcome_ledger_sha256": sha256(ledger_path),
        "opening_scores": matrix_rows,
    }
    matrix_path = DATA / "four-treatment-opening-score-matrix.json"
    write_json(matrix_path, matrix)

    rng = np.random.default_rng(398)
    draws = rng.integers(0, 512, size=(10_000, 512))
    contrasts = {
        "policy_FS_minus_SS": scores["FS"] - scores["SS"],
        "value_SF_minus_SS": scores["SF"] - scores["SS"],
        "FF_minus_SS": scores["FF"] - scores["SS"],
        "interaction_FF_minus_FS_minus_SF_plus_SS": scores["FF"]
        - scores["FS"]
        - scores["SF"]
        + scores["SS"],
    }
    estimates = {}
    for index, (label, values) in enumerate(contrasts.items()):
        quantiles = [0.0125, 0.9875] if index < 2 else [0.025, 0.975]
        bootstrap = values[draws].mean(axis=1)
        ci = [float(value) for value in np.quantile(bootstrap, quantiles)]
        mean = float(values.mean())
        classification = (
            "merits_follow_up_training_study"
            if index < 2 and mean >= 0.03 and ci[0] > 0
            else "supported_harmful_effect"
            if mean <= -0.03 and ci[1] < 0
            else "positive_but_threshold_not_met"
            if mean > 0 and index < 2
            else "uncertain"
        )
        estimates[label] = {
            "mean": mean,
            "interval": ci,
            "interval_level": "97.5%" if index < 2 else "95% descriptive",
            "quantiles": quantiles,
            "classification": classification,
        }
    outcomes: dict[str, Any] = {}
    for name, rows in games_by_treatment.items():
        by_seat = {}
        for seat in (0, 1):
            subset = [row for row in rows if row["challenger_player"] == seat]
            w = sum(row["winner"] == "challenger" for row in subset)
            d = sum(row["winner"] == "draw" for row in subset)
            losses = sum(row["winner"] == "current" for row in subset)
            by_seat[str(seat)] = {
                "games": len(subset),
                "wins": w,
                "draws": d,
                "losses": losses,
                "score": (w + 0.5 * d) / len(subset),
            }
        outcomes[name] = {
            "wins": sum(row["winner"] == "challenger" for row in rows),
            "draws": sum(row["winner"] == "draw" for row in rows),
            "losses": sum(row["winner"] == "current" for row in rows),
            "score": float(scores[name].mean()),
            "seat_scores": by_seat,
        }
    control_diff = (
        outcomes["SS"]["seat_scores"]["0"]["score"]
        - outcomes["SS"]["seat_scores"]["1"]["score"]
    )
    results = {
        "schema": "seed398-policy-value-composition-results-v1",
        "status": "valid",
        "registration_sha256": sha256(DATA / "registration.json"),
        "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
        "exclusion_proof_sha256": sha256(DATA / "opening-exclusion-proof.json"),
        "outcome_ledger_sha256": sha256(ledger_path),
        "opening_matrix_sha256": sha256(matrix_path),
        "total_games": len(ledger),
        "games_per_treatment": {
            name: len(rows) for name, rows in games_by_treatment.items()
        },
        "outcomes": outcomes,
        "SS_control_seat_score_difference_seat0_minus_seat1": control_diff,
        "SS_control_asymmetry_check": "reported descriptively; investigate if difference is large, no games dropped",
        "contrasts": estimates,
        "bootstrap": {
            "resamples": 10_000,
            "seed": 398,
            "cluster": "opening; all four treatment scores computed from both seats resampled together",
        },
        "exclusion_count": proof["excluded_state_count"],
        "decision": {
            "policy_follow_up": estimates["policy_FS_minus_SS"]["classification"]
            == "merits_follow_up_training_study",
            "value_follow_up": estimates["value_SF_minus_SS"]["classification"]
            == "merits_follow_up_training_study",
        },
    }
    write_json(DATA / "analysis.json", results)
    write_json(DATA / "per-opening-score-matrix.json", {"opening_scores": matrix_rows})
    lines = [
        "# Seed398 policy/value composition diagnostic",
        "",
        "All challenger treatments faced the unchanged seed455 opponent. Component outputs were composed through their own source evaluators; no weights were spliced.",
        "",
        "| Treatment | Score | Wins | Draws | Losses | Seat 0 score | Seat 1 score | Games |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, outcome in outcomes.items():
        lines.append(
            f"| {name} | {outcome['score']:.4f} | {outcome['wins']} | {outcome['draws']} | {outcome['losses']} | {outcome['seat_scores']['0']['score']:.4f} | {outcome['seat_scores']['1']['score']:.4f} | 1,024 |"
        )
    lines.extend(
        [
            "",
            "## Contrasts",
            "",
            "| Contrast | Mean | Interval | Level | Classification |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for label, result in estimates.items():
        lines.append(
            f"| {label} | {result['mean']:+.4f} | [{result['interval'][0]:+.4f}, {result['interval'][1]:+.4f}] | {result['interval_level']} | {result['classification']} |"
        )
    lines.extend(
        [
            "",
            f"SS seat-score difference (seat 0 minus seat 1): {control_diff:+.4f}.",
            "Primary components merit a follow-up training study only if the estimated effect is at least +0.03 and the 97.5% interval lower bound exceeds zero. Wide intervals are uncertainty, not equivalence.",
            "",
            "The complete validated per-game ledger is `validated-outcome-ledger.jsonl`; the 512-opening score matrix is `four-treatment-opening-score-matrix.json`. Recompute the bootstrap by rerunning `python -m ml.alphazero_lite.seed398_composition_diagnostic analyze` against the hash-bound registration and ledgers.",
        ]
    )
    (DATA / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "bind", "run", "analyze"))
    stage = parser.parse_args().stage
    {"register": register, "bind": bind, "run": run, "analyze": analyze}[stage]()


if __name__ == "__main__":
    main()
