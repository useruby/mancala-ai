"""Seed 624 deterministic, model-independent prospective state population.

The module has no model, search, solver, or training imports.  Production
generation is deliberately guarded by an explicit authorization record.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
import tracemalloc
from sys import getsizeof
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

SCHEMA = "seed624-prospective-population-v1"
MASTER_SEED = 62320261121
SEED_CONTRACT = "seed624-sha256-v1; big-endian u64 prefix; Python random.Random"
OPENING_INVALID_PREFIX_POLICY = "consume-reserve-index-skip-record-provenance-v1"
OPENING_RESERVE = 512
OPENING_TARGET = 256
GAMES_PER_OPENING = 8
PREFIX_ACTIONS = 8
MAX_CONTINUATION_ACTIONS = 200
STONE_TARGETS = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
STONE_BIN_WEIGHTS = {str(stones): 0.2 for stones in STONE_TARGETS}
BIN_ORDER = tuple(STONE_TARGETS)
MIN_OPENING_CLUSTERS = 128
MAX_SELECTION_CANDIDATES = 10_240
MAX_SELECTION_INPUT_BYTES = 128 * 1024 * 1024
MAX_SELECTION_NODES = 250_000
MAX_SELECTION_SECONDS = 30
SELECTION_TIME_CHECK_INTERVAL_NODES = 1
MAX_SELECTION_WORKING_BYTES = 64 * 1024 * 1024
EXECUTION_SCOPE = "model-independent-generation-and-screening-only"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "ml/alphazero_lite/seed624_prospective_benchmark.py",
    "ml/alphazero_lite/verify_seed624_prospective_benchmark.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/requirements-seed624-prospective.txt",
    "ml/alphazero_lite/seed628_authenticated_screening.py",
    "ml/alphazero_lite/verify_seed628_authenticated_screening.py",
    "ml/alphazero_lite/seed636_authorization.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deep_size_bytes(value: Any) -> int:
    """Return cycle-safe retained Python object size for a frozen input bundle."""
    pending = [value]
    visited: set[int] = set()
    total = 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in visited:
            continue
        visited.add(identity)
        total += getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return total


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def identities(state: dict[str, Any]) -> tuple[str, str]:
    """Match Seed 622 canonical JSON and production kalah_v3 float32 identity."""
    canonical_id = sha256(canonical_bytes(state))
    encoded = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return canonical_id, sha256(encoded.tobytes())


def runtime_identity() -> dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "prng": f"random.Random/MT19937; Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "optimizer": "seed624-dfs-bounded-v2",
    }


def current_source_hashes() -> dict[str, str]:
    return {name: sha256((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def authenticate_frozen_membership_evidence(
    freeze: dict[str, Any], *, caller_evidence_path: str | None = None
) -> dict[str, Any]:
    """Load evidence only through the repository-pinned Seed 627 trust chain."""
    if caller_evidence_path is not None:
        raise PermissionError("caller_controlled_membership_evidence_forbidden")
    from ml.alphazero_lite.seed628_authenticated_screening import (
        authenticate_snapshot_bundle,
        expected_provenance_hash,
        expected_snapshot_hashes,
    )

    evidence = authenticate_snapshot_bundle()
    if freeze.get("membership_snapshot_sha256") != expected_snapshot_hashes(evidence):
        raise PermissionError("execution_freeze_seed627_snapshot_hash_mismatch")
    if freeze.get("membership_snapshot_file_sha256") != {
        name: evidence[name]["file_sha256"]
        for name in ("B512", "S615", "protected_holdout")
    }:
        raise PermissionError("execution_freeze_seed627_snapshot_file_hash_mismatch")
    if (
        freeze.get("membership_source_manifest_sha256")
        != evidence["authentication"]["manifest_sha256"]
    ):
        raise PermissionError("execution_freeze_seed627_manifest_hash_mismatch")
    if freeze.get("source_provenance_sha256") != expected_provenance_hash(evidence):
        raise PermissionError("execution_freeze_seed627_provenance_hash_mismatch")
    return evidence


def derive_seed(
    master_seed: int, opening_index: int, trajectory_index: int, phase: str
) -> int:
    if not 0 <= opening_index < OPENING_RESERVE:
        raise ValueError("opening_index_outside_frozen_reserve")
    if trajectory_index < 0 or phase not in {"opening", "continuation"}:
        raise ValueError("invalid_seed_coordinates")
    message = canonical_bytes(
        {
            "domain": "useruby.mancala-ai/seed624/seed/v1",
            "master_seed": int(master_seed),
            "opening_index": int(opening_index),
            "trajectory_index": int(trajectory_index),
            "phase": phase,
        }
    )
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def standard_game() -> KalahGame:
    return KalahGame(
        pits=[4] * 12,
        captured_seeds=[0, 0],
        current_player=0,
    )


def _random_game(game: KalahGame, rng: random.Random, limit: int) -> list[int]:
    actions: list[int] = []
    for _ in range(limit):
        legal = game.possible_moves()
        if not legal:
            break
        action = rng.choice(legal)
        if not game.move(game.pit_index(action)):
            raise RuntimeError("rules_engine_rejected_its_legal_action")
        actions.append(action)
        if game.over():
            break
    return actions


class InvalidOpeningPrefix(ValueError):
    """A deterministic opening seed failed the frozen eight-action prefix rule."""

    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__(str(record.get("reason", "invalid_opening_prefix")))


def generate_opening(master_seed: int, opening_index: int) -> dict[str, Any]:
    seed = derive_seed(master_seed, opening_index, 0, "opening")
    game = standard_game()
    moves = _random_game(game, random.Random(seed), PREFIX_ACTIONS)
    terminal = game.over()
    if len(moves) != PREFIX_ACTIONS or terminal:
        reason = (
            "terminal_before_eight_actions"
            if terminal and len(moves) < PREFIX_ACTIONS
            else "terminal_at_eight_actions"
            if terminal
            else "prefix_action_count_mismatch"
        )
        state = game.to_state()
        canonical_id, float_id = identities(state)
        raise InvalidOpeningPrefix(
            {
                "opening_index": opening_index,
                "seed": seed,
                "status": "rejected_invalid_prefix",
                "reason": reason,
                "actions": moves,
                "terminal": terminal,
                "state": state,
                "canonical_identity": canonical_id,
                "production_float32_identity": float_id,
            }
        )
    canonical_id, float_id = identities(game.to_state())
    return {
        "opening_index": opening_index,
        "seed": seed,
        "state": game.to_state(),
        "canonical_identity": canonical_id,
        "production_float32_identity": float_id,
        "prefix_actions": moves,
    }


def choose_distinct_openings(
    opening_factory: Any,
    *,
    reserve: int,
    target: int,
    master_seed: int = MASTER_SEED,
    on_attempt=None,
) -> dict[str, Any]:
    """Consume each reserve index once; skip and record invalid/duplicate prefixes."""
    chosen: list[dict[str, Any]] = []
    canonical_seen: dict[str, int] = {}
    float_seen: dict[str, int] = {}
    attempts: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index in range(reserve):
        try:
            opening = opening_factory(index)
        except InvalidOpeningPrefix as error:
            attempt = dict(error.record)
            if (
                int(attempt.get("opening_index", -1)) != index
                or int(attempt.get("seed", -1))
                != derive_seed(master_seed, index, 0, "opening")
                or attempt.get("status") != "rejected_invalid_prefix"
            ):
                raise ValueError("invalid_opening_prefix_rejection_record_mismatch")
            attempts.append(attempt)
            rejections.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt, None)
            continue
        cid = opening["canonical_identity"]
        fid = opening["production_float32_identity"]
        seed = derive_seed(master_seed, index, 0, "opening")
        if cid in canonical_seen:
            attempt = {
                "opening_index": index,
                "seed": seed,
                "status": "rejected_duplicate",
                "reason": "canonical_identity_duplicate",
                "duplicate_of_opening_index": canonical_seen[cid],
                "canonical_identity": cid,
                "production_float32_identity": fid,
            }
            attempts.append(attempt)
            rejections.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt, None)
            continue
        if fid in float_seen:
            attempt = {
                "opening_index": index,
                "seed": seed,
                "status": "rejected_duplicate",
                "reason": "production_float32_identity_duplicate",
                "duplicate_of_opening_index": float_seen[fid],
                "canonical_identity": cid,
                "production_float32_identity": fid,
            }
            attempts.append(attempt)
            rejections.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt, None)
            continue
        canonical_seen[cid] = index
        float_seen[fid] = index
        chosen.append(opening)
        attempt = {
            "opening_index": index,
            "seed": seed,
            "status": "accepted",
            "canonical_identity": cid,
            "production_float32_identity": fid,
        }
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt, opening)
        if len(chosen) == target:
            return {
                "openings": chosen,
                "consumed_seed_indexes": list(range(index + 1)),
                "attempts": attempts,
                "rejections": rejections,
            }
    raise ShortfallError(
        {
            "schema": "seed624-generation-shortfall-v1",
            "reason": "opening_reserve_exhausted",
            "required": target,
            "available": len(chosen),
            "reserve": reserve,
            "consumed_seed_indexes": list(range(reserve)),
            "openings": chosen,
            "attempts": attempts,
            "rejections": rejections,
        }
    )


def _candidate_rank(
    master_seed: int, opening_id: str, game_id: str, stones: int, cid: str
) -> str:
    return sha256(
        canonical_bytes(
            {
                "domain": "useruby.mancala-ai/seed624/candidate-rank/v1",
                "master_seed": master_seed,
                "opening_id": opening_id,
                "game_id": game_id,
                "stone_bin": stones,
                "canonical_identity": cid,
            }
        )
    )


def generate_game(
    opening: dict[str, Any],
    master_seed: int,
    trajectory_index: int,
    *,
    on_action=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opening_index = int(opening["opening_index"])
    game_id = f"o{opening_index:04d}-g{trajectory_index:02d}"
    game = KalahGame.from_state(opening["state"])
    continuation_seed = derive_seed(
        master_seed, opening_index, trajectory_index, "continuation"
    )
    rng = random.Random(continuation_seed)
    trajectory: list[int] = []
    visits_by_bin: dict[int, list[dict[str, Any]]] = {}
    for ply in range(MAX_CONTINUATION_ACTIONS):
        if game.over():
            break
        legal = game.possible_moves()
        stones = sum(game.pits)
        if stones in STONE_TARGETS and len(legal) >= 2:
            state = game.to_state()
            canonical_id, float_id = identities(state)
            visits_by_bin.setdefault(stones, []).append(
                {
                    "state": state,
                    "canonical_identity": canonical_id,
                    "production_float32_identity": float_id,
                    "active_pit_stones": stones,
                    "legal_actions": legal,
                    "side_to_move": game.current_player,
                    "opening_index": opening_index,
                    "opening_identity": opening["canonical_identity"],
                    "game_id": game_id,
                    "continuation_ply": ply,
                    "trajectory": list(trajectory),
                    "rank": _candidate_rank(
                        master_seed,
                        opening["canonical_identity"],
                        game_id,
                        stones,
                        canonical_id,
                    ),
                }
            )
        action = rng.choice(legal)
        if not game.move(game.pit_index(action)):
            raise RuntimeError("rules_engine_rejected_its_legal_action")
        trajectory.append(action)
        if on_action is not None:
            on_action(
                {
                    "game_id": game_id,
                    "opening_index": opening_index,
                    "trajectory_index": trajectory_index,
                    "seed": continuation_seed,
                    "continuation_ply": ply,
                    "action": action,
                }
            )
    candidates = []
    for stones, visits in sorted(visits_by_bin.items()):
        visits.sort(key=lambda row: (row["rank"], row["canonical_identity"]))
        selected_visit = dict(visits[0])
        selected_visit["within_game_bin_visits"] = [
            {
                "canonical_identity": row["canonical_identity"],
                "production_float32_identity": row["production_float32_identity"],
                "continuation_ply": row["continuation_ply"],
                "rank": row["rank"],
            }
            for row in visits
        ]
        candidates.append(selected_visit)
    terminal = game.over()
    return {
        "game_id": game_id,
        "opening_index": opening_index,
        "opening_identity": opening["canonical_identity"],
        "seed": continuation_seed,
        "actions": trajectory,
        "full_actions": list(opening["prefix_actions"]) + trajectory,
        "terminal": terminal,
        "action_limit_reached": (
            len(trajectory) == MAX_CONTINUATION_ACTIONS and not terminal
        ),
        "final_state": game.to_state(),
    }, candidates


class ShortfallError(ValueError):
    """A hard generation or selection gate failed; carries machine-readable data."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(json.dumps(report, sort_keys=True))


def _write_json_create_only(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()


def _write_jsonl_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row) + b"\n")
        stream.flush()


def _shortfall_classification(report: dict[str, Any]) -> str:
    reason = str(report.get("reason", ""))
    if reason in {"opening_reserve_exhausted", "hard_supply"}:
        return "insufficient_candidate_supply"
    if reason in {"aggregate_cluster_capacity", "constraints_infeasible"}:
        return "mathematical_infeasibility"
    if "resource_limit" in reason or reason in {
        "candidate_count_resource_limit",
        "input_memory_resource_limit",
    }:
        return "resource_exhaustion"
    return "verification_failure"


def _write_failure_record(
    output: Path,
    *,
    stage: str,
    classification: str,
    reason: dict[str, Any],
) -> None:
    _write_json_create_only(
        output / "failure.json",
        {
            "schema": "seed624-population-generation-failure-v1",
            "status": "failed",
            "stage": stage,
            "classification": classification,
            "reason": reason,
            "qualified_population": False,
        },
    )


def _run_selection_with_evidence(
    output: Path, screened: list[dict[str, Any]]
) -> dict[str, Any]:
    eligible = [
        row
        for row in screened
        if row.get("classification")
        == "known_exposure_nonmatch_inherited_PE4_unresolved"
    ]
    supply = Counter(int(row["active_pit_stones"]) for row in eligible)
    diagnostics = {
        "schema": "seed624-selection-input-diagnostics-v1",
        "candidate_count": len(screened),
        "eligible_candidate_count": len(eligible),
        "eligible_supply": {str(stones): supply[stones] for stones in BIN_ORDER},
        "hard_minimum_supply": {
            str(stones): 2 * target for stones, target in STONE_TARGETS.items()
        },
        "preferred_four_x_supply": {
            str(stones): supply[stones] >= 4 * target
            for stones, target in STONE_TARGETS.items()
        },
        "candidate_inventory_sha256": sha256(canonical_bytes(screened)),
        "selection_contract": {
            "targets": {str(stones): count for stones, count in STONE_TARGETS.items()},
            "minimum_openings": MIN_OPENING_CLUSTERS,
            "max_per_opening": 4,
            "max_per_game": 2,
            "workers": 1,
            "optimizer": "seed624-dfs-bounded-v2",
            "candidate_limit": MAX_SELECTION_CANDIDATES,
            "input_memory_limit_bytes": MAX_SELECTION_INPUT_BYTES,
            "node_limit": MAX_SELECTION_NODES,
            "runtime_limit_seconds": MAX_SELECTION_SECONDS,
            "working_memory_limit_bytes": MAX_SELECTION_WORKING_BYTES,
        },
    }
    _write_json_create_only(output / "selection-input-diagnostics.json", diagnostics)
    try:
        selection = select_cohort(screened)
    except ShortfallError as error:
        _write_json_create_only(
            output / "selection-diagnostics.json",
            {
                "schema": "seed624-selection-shortfall-v1",
                "status": "shortfall",
                "failure_classification": _shortfall_classification(error.report),
                "input": diagnostics,
                "selector_report": error.report,
                "qualified_population": False,
            },
        )
        _write_failure_record(
            output,
            stage="constrained_selection",
            classification=_shortfall_classification(error.report),
            reason=error.report,
        )
        raise
    except BaseException as error:
        failure = {
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        _write_json_create_only(
            output / "selection-diagnostics.json",
            {
                "schema": "seed624-selection-failure-v1",
                "status": "failed",
                "input": diagnostics,
                "failure": failure,
                "qualified_population": False,
            },
        )
        _write_failure_record(
            output,
            stage="constrained_selection",
            classification="verification_failure",
            reason=failure,
        )
        raise
    _write_json_create_only(
        output / "selection-diagnostics.json",
        {
            "schema": "seed624-selection-result-v1",
            "status": "selected",
            "input": diagnostics,
            "selection": selection,
        },
    )
    return selection


def _persist_screened_stages_and_select(
    output: Path,
    screened_candidates: list[dict[str, Any]],
    *,
    inventories_already_persisted: bool = False,
) -> dict[str, Any]:
    if not inventories_already_persisted:
        _write_jsonl_create_only(
            output / "screened-candidates.jsonl", screened_candidates
        )
        _write_jsonl_create_only(
            output / "exclusions.jsonl",
            [
                row
                for row in screened_candidates
                if row["classification"]
                != "known_exposure_nonmatch_inherited_PE4_unresolved"
            ],
        )
    return _run_selection_with_evidence(output, screened_candidates)


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first ranked row; reject canonical OR encoded collisions globally."""
    ordered = sorted(
        candidates, key=lambda row: (row["rank"], row["canonical_identity"])
    )
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    result: list[dict[str, Any]] = []
    by_canonical: dict[str, dict[str, Any]] = {}
    by_float: dict[str, dict[str, Any]] = {}
    for row in ordered:
        cid, fid = row["canonical_identity"], row["production_float32_identity"]
        if cid in canonical_seen:
            by_canonical[cid].setdefault("occurrences", []).append(
                {
                    key: row[key]
                    for key in ("opening_index", "game_id", "continuation_ply", "rank")
                }
            )
            continue
        if fid in float_seen:
            by_float[fid].setdefault(
                "float32_identity_collision_occurrences", []
            ).append(
                {
                    "canonical_identity": cid,
                    "opening_index": row["opening_index"],
                    "game_id": row["game_id"],
                    "continuation_ply": row["continuation_ply"],
                    "rank": row["rank"],
                }
            )
            continue
        canonical_seen.add(cid)
        float_seen.add(fid)
        item = {
            **row,
            "occurrences": [
                {
                    key: row[key]
                    for key in ("opening_index", "game_id", "continuation_ply", "rank")
                }
            ],
        }
        by_canonical[cid] = item
        by_float[fid] = item
        result.append(item)
    return result


def screen_known_exposure(
    candidates: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    expected_snapshot_hashes: dict[str, str] | None = None,
    expected_provenance_hash: str | None = None,
    on_screened=None,
) -> list[dict[str, Any]]:
    """Fail closed unless every required identity inventory is authenticated."""
    required = ("B512", "S615", "protected_holdout")
    inventories: dict[str, tuple[set[str], set[str]]] = {}
    evidence_hashes: dict[str, str] = {}
    for name in required:
        source = evidence.get(name)
        if not isinstance(source, dict) or source.get("authenticated") is not True:
            raise ValueError(f"invalid_or_unauthenticated_membership:{name}")
        rows = source.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing_membership_rows:{name}")
        actual_hash = sha256(canonical_bytes(rows))
        if (
            source.get("sha256") != actual_hash
            or expected_snapshot_hashes is None
            or expected_snapshot_hashes.get(name) != actual_hash
        ):
            raise ValueError(f"membership_hash_mismatch:{name}")
        canonical_ids: set[str] = set()
        float_ids: set[str] = set()
        for row in rows:
            state = row.get("canonical_state")
            if not isinstance(state, dict):
                raise ValueError(f"membership_state_missing:{name}")
            cid, fid = identities(state)
            if cid != row.get("canonical_identity") or fid != row.get(
                "production_float32_identity"
            ):
                raise ValueError(f"membership_identity_mismatch:{name}")
            canonical_ids.add(cid)
            float_ids.add(fid)
        inventories[name] = canonical_ids, float_ids
        evidence_hashes[name] = actual_hash

    provenance = evidence.get("source_provenance", {})
    provenance_authenticated = expected_provenance_hash is not None
    if provenance_authenticated:
        if not provenance:
            raise ValueError("source_provenance_missing")
        if provenance.get("authenticated") is not True:
            raise ValueError("source_provenance_unauthenticated")
        provenance_payload = {
            key: value
            for key, value in provenance.items()
            if key not in {"authenticated", "sha256"}
        }
        actual_provenance_hash = sha256(canonical_bytes(provenance_payload))
        if (
            provenance.get("sha256") != actual_provenance_hash
            or expected_provenance_hash != actual_provenance_hash
        ):
            raise ValueError("source_provenance_hash_mismatch")
    elif provenance.get("authenticated") is True:
        raise ValueError("source_provenance_hash_required")
    provenance_crosswalkable = (
        provenance_authenticated and provenance.get("crosswalk_status") == "complete"
    )
    if provenance_authenticated and provenance.get("crosswalk_status") not in {
        "complete",
        "authenticated_non_crosswalkable",
    }:
        raise ValueError("source_provenance_crosswalk_status_invalid")
    if (
        provenance.get("crosswalk_status") == "authenticated_non_crosswalkable"
        and provenance.get("crosswalk_rows") != []
    ):
        raise ValueError("source_provenance_crosswalk_claim_invalid")
    if provenance_crosswalkable:
        for name in required:
            if not isinstance(provenance.get(name), list):
                raise ValueError(f"source_provenance_incomplete:{name}")
    screened = []
    for candidate in candidates:
        cid, fid = (
            candidate["canonical_identity"],
            candidate["production_float32_identity"],
        )
        hits = {
            name: cid in ids or fid in floats
            for name, (ids, floats) in inventories.items()
        }
        provenance_match: bool | None = False if provenance_crosswalkable else None
        if provenance_crosswalkable:
            for name in required:
                sources = provenance.get(name)
                if any(
                    item.get("opening_identity") == candidate["opening_identity"]
                    or item.get("game_id") == candidate["game_id"]
                    for item in sources
                ):
                    provenance_match = True
                    break
        confirmed = any(hits.values()) or provenance_match
        classification = (
            "confirmed_known_exposure"
            if confirmed
            else "known_exposure_nonmatch_inherited_PE4_unresolved"
            if provenance_authenticated
            else "invalid_or_insufficiently_authenticated_evidence"
        )
        screened_row = {
            **candidate,
            "exposure": hits,
            "source_provenance_match": provenance_match,
            "source_provenance_status": provenance.get(
                "crosswalk_status", "missing_or_unauthenticated"
            ),
            "classification": classification,
            "membership_evidence_sha256": evidence_hashes,
            "inherited_PE4_exposure": "unresolved",
        }
        screened.append(screened_row)
        if on_screened is not None:
            on_screened(screened_row)
    return screened


def select_cohort(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    tracing_was_active = tracemalloc.is_tracing()
    if not tracing_was_active:
        tracemalloc.start()
    try:
        return _select_cohort_bounded(candidates)
    finally:
        if not tracing_was_active:
            tracemalloc.stop()


def _select_cohort_bounded(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pinned deterministic rank-first constrained selection for frozen inventories.

    This exact search visits candidates in the declared within-bin rank order.
    It returns the first feasible lexicographic rank vector and has a hard
    single-worker, node-, time-, candidate-, and memory-bounded exact search.
    """
    if len(candidates) > MAX_SELECTION_CANDIDATES:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "candidate_count_resource_limit",
                "available": len(candidates),
                "limit": MAX_SELECTION_CANDIDATES,
            }
        )
    input_bytes = deep_size_bytes(candidates)
    if input_bytes > MAX_SELECTION_INPUT_BYTES:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "input_memory_resource_limit",
                "input_bytes": input_bytes,
                "limit_bytes": MAX_SELECTION_INPUT_BYTES,
            }
        )
    eligible = [
        row
        for row in candidates
        if row.get("classification")
        == "known_exposure_nonmatch_inherited_PE4_unresolved"
    ]
    by_bin = {stones: [] for stones in BIN_ORDER}
    for row in eligible:
        stones = int(row["active_pit_stones"])
        if stones in by_bin:
            by_bin[stones].append(row)
    supply = {stones: len(rows) for stones, rows in by_bin.items()}
    hard_supply = {stones: 2 * target for stones, target in STONE_TARGETS.items()}
    short = {
        stones: {"required": hard_supply[stones], "available": supply[stones]}
        for stones in BIN_ORDER
        if supply[stones] < hard_supply[stones]
    }
    if short:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "hard_supply",
                "bins": short,
            }
        )

    total_target = sum(STONE_TARGETS.values())
    game_capacity = sum(
        min(2, count)
        for count in Counter(str(row["game_id"]) for row in eligible).values()
    )
    opening_capacity = sum(
        min(4, count)
        for count in Counter(int(row["opening_index"]) for row in eligible).values()
    )
    distinct_games = len({str(row["game_id"]) for row in eligible})
    distinct_openings = len({int(row["opening_index"]) for row in eligible})
    if (
        game_capacity < total_target
        or opening_capacity < total_target
        or distinct_games < (total_target + 1) // 2
        or distinct_openings < MIN_OPENING_CLUSTERS
    ):
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "aggregate_cluster_capacity",
                "target": total_target,
                "game_capacity": game_capacity,
                "opening_capacity": opening_capacity,
                "distinct_games": distinct_games,
                "distinct_openings": distinct_openings,
            }
        )

    # Exact rank-first depth-first feasibility search with hard node, time,
    # candidate-count, and traced-working-memory ceilings. No memo table is
    # retained, bounding auxiliary storage to counters plus the chosen path.
    start = time.monotonic()
    budget_seconds = MAX_SELECTION_SECONDS
    ordered = {
        stones: sorted(rows, key=lambda row: (row["rank"], row["canonical_identity"]))
        for stones, rows in by_bin.items()
    }
    nodes = 0
    chosen: list[dict[str, Any]] = []
    game_counts: Counter[str] = Counter()
    opening_counts: Counter[int] = Counter()
    opening_set: set[int] = set()
    peak_working_bytes = 0

    def search(bin_index: int, start_index: int) -> bool:
        nonlocal nodes, peak_working_bytes
        nodes += 1
        if nodes > MAX_SELECTION_NODES:
            raise TimeoutError("node_limit")
        if (
            nodes % SELECTION_TIME_CHECK_INTERVAL_NODES == 0
            and time.monotonic() - start > budget_seconds
        ):
            raise TimeoutError
        if nodes % 128 == 0:
            _current_bytes, peak_working_bytes = tracemalloc.get_traced_memory()
            if peak_working_bytes > MAX_SELECTION_WORKING_BYTES:
                raise MemoryError
        if bin_index == len(BIN_ORDER):
            return len(opening_set) >= MIN_OPENING_CLUSTERS
        stones = BIN_ORDER[bin_index]
        target = STONE_TARGETS[stones]
        in_bin = sum(row["active_pit_stones"] == stones for row in chosen)
        remaining = target - in_bin
        rows = ordered[stones]
        if remaining == 0:
            return search(bin_index + 1, 0)
        available_count = sum(
            1
            for index in range(start_index, len(rows))
            if game_counts[rows[index]["game_id"]] < 2
            and opening_counts[int(rows[index]["opening_index"])] < 4
        )
        if available_count < remaining:
            return False
        # Preserve rank order and include the earliest rank whose inclusion
        # admits a complete feasible suffix. Canonical identity breaks rank ties.
        skipped_count = 0
        for index in range(start_index, len(rows)):
            row = rows[index]
            if (
                game_counts[row["game_id"]] >= 2
                or opening_counts[int(row["opening_index"])] >= 4
            ):
                continue
            chosen.append(row)
            game_counts[row["game_id"]] += 1
            opening = int(row["opening_index"])
            opening_counts[opening] += 1
            was_new = opening not in opening_set
            opening_set.add(opening)
            if search(bin_index, index + 1):
                return True
            if was_new and opening_counts[opening] == 1:
                opening_set.remove(opening)
            opening_counts[opening] -= 1
            game_counts[row["game_id"]] -= 1
            chosen.pop()
            skipped_count += 1
            if available_count - skipped_count < remaining:
                break
        return False

    try:
        feasible = search(0, 0)
    except (TimeoutError, MemoryError) as error:
        limit = (
            "working_memory_bytes"
            if isinstance(error, MemoryError)
            else str(error) or "time_seconds"
        )
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "optimizer_resource_limit",
                "limit": limit,
                "nodes": nodes,
                "elapsed_seconds": time.monotonic() - start,
                "peak_working_bytes": peak_working_bytes,
                "runtime_limit_seconds": budget_seconds,
                "node_limit": MAX_SELECTION_NODES,
                "time_check_interval_nodes": SELECTION_TIME_CHECK_INTERVAL_NODES,
                "working_memory_limit_bytes": MAX_SELECTION_WORKING_BYTES,
            }
        ) from None
    if not feasible:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "constraints_infeasible",
                "supply": supply,
                "distinct_openings": distinct_openings,
                "nodes": nodes,
                "elapsed_seconds": time.monotonic() - start,
                "peak_working_bytes": peak_working_bytes,
            }
        )
    selected = sorted(
        chosen,
        key=lambda row: (
            row["active_pit_stones"],
            row["rank"],
            row["canonical_identity"],
        ),
    )
    selected_inventory_sha256 = sha256(canonical_bytes(selected))
    counts = Counter(int(row["active_pit_stones"]) for row in selected)
    openings = Counter(int(row["opening_index"]) for row in selected)
    games = Counter(str(row["game_id"]) for row in selected)
    return {
        "schema": "seed624-selected-cohort-v1",
        "targets": STONE_TARGETS,
        "selection_contract": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity; bounded exhaustive feasibility; seed624-dfs-bounded-v2",
        "optimizer": {
            "name": "seed624-dfs-bounded-v2",
            "version": 2,
            "workers": 1,
            "runtime_limit_seconds": budget_seconds,
            "candidate_limit": MAX_SELECTION_CANDIDATES,
            "input_memory_limit_bytes": MAX_SELECTION_INPUT_BYTES,
            "input_bytes": input_bytes,
            "node_limit": MAX_SELECTION_NODES,
            "time_check_interval_nodes": SELECTION_TIME_CHECK_INTERVAL_NODES,
            "working_memory_limit_bytes": MAX_SELECTION_WORKING_BYTES,
            "peak_working_bytes": peak_working_bytes,
            "elapsed_seconds": time.monotonic() - start,
            "nodes": nodes,
        },
        "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
        "stone_bin_weights": STONE_BIN_WEIGHTS,
        "eligible_supply": supply,
        "preferred_four_x_supply": {
            str(stones): supply[stones] >= 4 * target
            for stones, target in STONE_TARGETS.items()
        },
        "counts_by_stone": dict(counts),
        "selected_inventory_sha256": selected_inventory_sha256,
        "minimum_openings": MIN_OPENING_CLUSTERS,
        "distinct_openings": len(openings),
        "max_per_opening": max(openings.values()),
        "max_per_game": max(games.values()),
        "candidate_inventory_sha256": sha256(canonical_bytes(eligible)),
        "states": selected,
    }


def generate_population(authorization: dict[str, Any], output: Path) -> dict[str, Any]:
    """Generate only from a frozen attempt with an owner-merged registration PR."""
    from ml.alphazero_lite.seed636_authorization import validate_attempt

    attempt_dir = Path(authorization.get("attempt_dir", ""))
    manifest, evidence = validate_attempt(attempt_dir)
    output = output.resolve()
    declared_outputs = {
        (attempt_dir / item).resolve() for item in manifest.get("output_paths", [])
    }
    if output not in declared_outputs or not output.is_relative_to(
        attempt_dir.resolve()
    ):
        raise PermissionError("lifecycle_output_path_binding_mismatch")
    freeze = manifest.get("configuration", {}).get("seed624_freeze")
    if not isinstance(freeze, dict):
        raise PermissionError("seed624_generation_configuration_missing")
    freeze_raw = canonical_bytes(freeze)
    registration_pr_authorization = evidence["authorization"]
    owner_identity = registration_pr_authorization["merged_by"]
    attempt_manifest_raw = (attempt_dir / "manifest.json").read_bytes()
    lifecycle_auth = {
        "attempt_manifest_sha256": sha256(attempt_manifest_raw),
        "attempt_manifest_raw_utf8": attempt_manifest_raw.decode("utf-8"),
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "registration_sha256": sha256(
            (attempt_dir.parents[3] / ".experiment" / "registration.json").read_bytes()
        ),
        "registration_raw_utf8": (
            attempt_dir.parents[3] / ".experiment" / "registration.json"
        ).read_text(),
    }
    expected_sources = freeze.get("source_file_sha256")
    expected_runtime = freeze.get("runtime")
    rules_hash = sha256((ROOT / "ml/alphazero_lite/kalah_rules.py").read_bytes())
    encoder_hash = sha256((ROOT / "ml/alphazero_lite/input_encodings.py").read_bytes())
    generation_contract = freeze.get("generation")
    selection_contract = freeze.get("selection")
    production_resources = freeze.get("production_resource_limits")
    production_resources_valid = isinstance(production_resources, dict) and all(
        type(production_resources.get(name)) is int and production_resources[name] > 0
        for name in (
            "wall_seconds",
            "cpu_seconds",
            "process_rss_bytes",
            "execution_concurrency",
            "output_and_temporary_storage_bytes",
        )
    )
    resource_values = (
        production_resources if isinstance(production_resources, dict) else {}
    )
    registered_budget = evidence["registration"].get("resource_budgets", {})
    registered_resources_match = (
        resource_values.get("wall_seconds")
        == registered_budget.get("wall_time_limit_seconds")
        and resource_values.get("cpu_seconds")
        == registered_budget.get("cpu_time_limit_seconds")
        and resource_values.get("process_rss_bytes")
        == registered_budget.get("process_rss_limit_bytes")
        and resource_values.get("execution_concurrency")
        == registered_budget.get("execution_concurrency")
        and resource_values.get("output_and_temporary_storage_bytes")
        == registered_budget.get("output_and_temporary_storage_limit_bytes")
    )
    lifecycle_budget = manifest.get("resource_budget", {})
    additional_limits = lifecycle_budget.get("additional_limits", {})
    resource_bindings_match = False
    if production_resources_valid:
        resource_bindings_match = (
            resource_values["wall_seconds"]
            == lifecycle_budget.get("wall_time_limit_seconds")
            and resource_values["cpu_seconds"] == additional_limits.get("cpu_seconds")
            and resource_values["process_rss_bytes"]
            == additional_limits.get("process_rss_bytes")
            and resource_values["execution_concurrency"] == manifest.get("worker_count")
            and resource_values["output_and_temporary_storage_bytes"]
            == additional_limits.get("output_and_temporary_storage_bytes")
        )
    if (
        freeze.get("schema") != "seed624-execution-freeze-v1"
        or freeze.get("status") != "frozen"
        or freeze.get("execution_scope") != EXECUTION_SCOPE
        or manifest.get("configuration", {}).get("execution_scope") != EXECUTION_SCOPE
        or manifest.get("configuration", {}).get("registration_sha256")
        != lifecycle_auth["registration_sha256"]
        or str(output) not in freeze.get("create_only_output_paths", [])
        or freeze.get("git_commit") != current_git_commit()
        or not isinstance(expected_sources, dict)
        or expected_sources != current_source_hashes()
        or expected_runtime != runtime_identity()
        or freeze.get("rules_identity")
        != {
            "rule_id": "historical-kalah-rules",
            "rule_version": "repository-kalah-rules-v1",
            "source_sha256": rules_hash,
        }
        or freeze.get("encoder_identity")
        != {"encoder": "kalah_v3", "source_sha256": encoder_hash}
        or generation_contract
        != {
            "master_seed": MASTER_SEED,
            "seed_contract": SEED_CONTRACT,
            "opening_reserve": OPENING_RESERVE,
            "invalid_prefix_policy": OPENING_INVALID_PREFIX_POLICY,
            "opening_target": OPENING_TARGET,
            "prefix_actions": PREFIX_ACTIONS,
            "games_per_opening": GAMES_PER_OPENING,
            "continuation_action_limit": MAX_CONTINUATION_ACTIONS,
        }
        or selection_contract
        != {
            "targets": {str(key): value for key, value in STONE_TARGETS.items()},
            "max_per_game": 2,
            "max_per_opening": 4,
            "minimum_openings": MIN_OPENING_CLUSTERS,
            "objective": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity",
            "optimizer": "seed624-dfs-bounded-v2",
            "candidate_limit": MAX_SELECTION_CANDIDATES,
            "runtime_limit_seconds": MAX_SELECTION_SECONDS,
            "node_limit": MAX_SELECTION_NODES,
            "time_check_interval_nodes": SELECTION_TIME_CHECK_INTERVAL_NODES,
            "input_memory_limit_bytes": MAX_SELECTION_INPUT_BYTES,
            "working_memory_limit_bytes": MAX_SELECTION_WORKING_BYTES,
            "workers": 1,
        }
        or not resource_bindings_match
        or not registered_resources_match
        or freeze.get("sampling_contract")
        != "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim"
        or freeze.get("stone_bin_weights") != STONE_BIN_WEIGHTS
        or freeze.get("output_schema") != SCHEMA
    ):
        raise PermissionError("execution_freeze_authorization_binding_mismatch")
    if output.exists():
        raise FileExistsError("production_artifact_path_must_be_create_only")
    evidence = authenticate_frozen_membership_evidence(
        freeze,
        caller_evidence_path=authorization.get("membership_evidence_path"),
    )
    output.mkdir(parents=True)
    _write_json_create_only(
        output / "generation-configuration.json",
        {
            "master_seed": MASTER_SEED,
            "seed_contract": SEED_CONTRACT,
            "opening_reserve": OPENING_RESERVE,
            "opening_target": OPENING_TARGET,
            "games_per_opening": GAMES_PER_OPENING,
            "continuation_action_limit": MAX_CONTINUATION_ACTIONS,
            "freeze_sha256": sha256(freeze_raw),
        },
    )
    _write_json_create_only(output / "authenticated-membership-evidence.json", evidence)
    try:
        with (
            (output / "opening-attempts.jsonl").open("xb") as attempts_stream,
            (output / "accepted-openings.jsonl").open("xb") as openings_stream,
        ):

            def record_opening_attempt(
                attempt: dict[str, Any], opening: dict[str, Any] | None
            ) -> None:
                attempts_stream.write(canonical_bytes(attempt) + b"\n")
                attempts_stream.flush()
                if opening is not None:
                    openings_stream.write(canonical_bytes(opening) + b"\n")
                    openings_stream.flush()

            opening_selection = choose_distinct_openings(
                lambda index: generate_opening(MASTER_SEED, index),
                reserve=OPENING_RESERVE,
                target=OPENING_TARGET,
                on_attempt=record_opening_attempt,
            )
    except ShortfallError as error:
        report = error.report
        _write_json_create_only(
            output / "opening-attempts.json", report.get("attempts", [])
        )
        _write_json_create_only(
            output / "opening-rejections.json", report.get("rejections", [])
        )
        _write_json_create_only(output / "openings.json", report.get("openings", []))
        _write_json_create_only(
            output / "opening-selection-shortfall.json",
            {
                "schema": "seed624-opening-selection-shortfall-v1",
                "report": report,
            },
        )
        _write_failure_record(
            output,
            stage="opening_generation",
            classification=_shortfall_classification(report),
            reason=report,
        )
        raise
    except BaseException as error:
        _write_failure_record(
            output,
            stage="opening_generation",
            classification="verification_failure",
            reason={"exception_type": type(error).__name__, "message": str(error)},
        )
        raise
    openings = opening_selection["openings"]
    _write_json_create_only(
        output / "opening-attempts.json", opening_selection["attempts"]
    )
    _write_json_create_only(
        output / "opening-rejections.json", opening_selection["rejections"]
    )
    _write_json_create_only(output / "openings.json", openings)
    games: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    action_count = [0]
    try:
        with (
            (output / "games.jsonl").open("xb") as games_stream,
            (output / "candidate-occurrences.jsonl").open("xb") as occurrence_stream,
            (output / "continuation-action-stream.jsonl").open("xb") as actions_stream,
        ):

            def record_continuation_action(action_record: dict[str, Any]) -> None:
                actions_stream.write(canonical_bytes(action_record) + b"\n")
                actions_stream.flush()
                action_count[0] += 1

            for opening in openings:
                for trajectory_index in range(GAMES_PER_OPENING):
                    game, occurrences = generate_game(
                        opening,
                        MASTER_SEED,
                        trajectory_index,
                        on_action=record_continuation_action,
                    )
                    game["opening_prefix_actions"] = opening["prefix_actions"]
                    games.append(game)
                    candidates.extend(occurrences)
                    games_stream.write(canonical_bytes(game) + b"\n")
                    for occurrence in occurrences:
                        occurrence_stream.write(canonical_bytes(occurrence) + b"\n")
                    games_stream.flush()
                    occurrence_stream.flush()
    except BaseException as error:
        _write_failure_record(
            output,
            stage="trajectory_generation",
            classification="verification_failure",
            reason={
                "exception_type": type(error).__name__,
                "message": str(error),
                "completed_games": len(games),
                "recorded_candidate_occurrences": len(candidates),
                "recorded_continuation_actions": action_count[0],
            },
        )
        raise
    try:
        unique = deduplicate_candidates(candidates)
    except BaseException as error:
        _write_failure_record(
            output,
            stage="candidate_deduplication",
            classification="verification_failure",
            reason={"exception_type": type(error).__name__, "message": str(error)},
        )
        raise
    _write_jsonl_create_only(output / "candidate-inventory.jsonl", unique)
    try:
        with (
            (output / "screened-candidates.jsonl").open("xb") as screened_stream,
            (output / "exclusions.jsonl").open("xb") as exclusions_stream,
        ):

            def record_screening_result(row: dict[str, Any]) -> None:
                serialized = canonical_bytes(row) + b"\n"
                screened_stream.write(serialized)
                screened_stream.flush()
                if (
                    row["classification"]
                    != "known_exposure_nonmatch_inherited_PE4_unresolved"
                ):
                    exclusions_stream.write(serialized)
                    exclusions_stream.flush()

            screened = screen_known_exposure(
                unique,
                evidence,
                expected_snapshot_hashes=freeze.get("membership_snapshot_sha256"),
                expected_provenance_hash=freeze.get("source_provenance_sha256"),
                on_screened=record_screening_result,
            )
    except BaseException as error:
        _write_failure_record(
            output,
            stage="membership_screening",
            classification="verification_failure",
            reason={"exception_type": type(error).__name__, "message": str(error)},
        )
        raise
    selection = _persist_screened_stages_and_select(
        output, screened, inventories_already_persisted=True
    )
    bundle = {
        "schema": SCHEMA,
        "authorization": {
            "project_owner_identity": owner_identity,
            "registration_pr_authorization": registration_pr_authorization,
            "execution_freeze_sha256": sha256(freeze_raw),
            "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
            "lifecycle": lifecycle_auth,
        },
        "generation": {
            "master_seed": MASTER_SEED,
            "seed_contract": SEED_CONTRACT,
            "invalid_prefix_policy": OPENING_INVALID_PREFIX_POLICY,
            "opening_seeds_consumed": opening_selection["consumed_seed_indexes"],
            "opening_attempts": opening_selection["attempts"],
            "opening_rejections": opening_selection["rejections"],
        },
        "source_identity": {
            "git_commit": current_git_commit(),
            "runtime": runtime_identity(),
            "files": {
                name: {"sha256": digest, "source": (ROOT / name).read_text()}
                for name, digest in current_source_hashes().items()
            },
        },
        "opening_inventory_sha256": sha256(canonical_bytes(openings)),
        "opening_attempts_sha256": sha256(
            canonical_bytes(opening_selection["attempts"])
        ),
        "game_inventory_sha256": sha256(canonical_bytes(games)),
        "candidate_inventory_sha256": sha256(canonical_bytes(screened)),
        "aggregate_counts": {
            "openings": len(openings),
            "opening_seed_indexes_consumed": len(
                opening_selection["consumed_seed_indexes"]
            ),
            "invalid_opening_prefixes": sum(
                row["status"] == "rejected_invalid_prefix"
                for row in opening_selection["rejections"]
            ),
            "duplicate_openings": sum(
                row["status"] == "rejected_duplicate"
                for row in opening_selection["rejections"]
            ),
            "games": len(games),
            "candidates": len(screened),
            "confirmed_known_exposure": sum(
                row["classification"] == "confirmed_known_exposure" for row in screened
            ),
            "known_exposure_nonmatch_inherited_PE4_unresolved": sum(
                row["classification"]
                == "known_exposure_nonmatch_inherited_PE4_unresolved"
                for row in screened
            ),
            "invalid_or_insufficiently_authenticated_evidence": sum(
                row["classification"]
                == "invalid_or_insufficiently_authenticated_evidence"
                for row in screened
            ),
        },
        "openings": openings,
        "games": games,
        "candidates": screened,
        "selection": selection,
        "evidence": evidence,
    }
    _write_json_create_only(output / "population.json", bundle)
    with (output / "population.sha256").open("x", encoding="ascii") as stream:
        stream.write(sha256((output / "population.json").read_bytes()) + "\n")
    return {
        "schema": SCHEMA,
        "openings": len(openings),
        "games": len(games),
        "eligible_candidates": len(screened),
        "selected": len(selection["states"]),
        "output": str(output),
    }


def proposed_freeze_schema() -> dict[str, Any]:
    """Return a schema template, never a production-ready execution freeze."""
    return {
        "schema": "seed624-proposed-execution-freeze-v1",
        "status": "proposal_not_ready_or_approved",
        "required": {
            "git_commit": None,
            "source_file_sha256": {name: None for name in SOURCE_FILES},
            "independent_verifier_sha256": None,
            "rules_identity": {
                "rule_id": "historical-kalah-rules",
                "rule_version": "repository-kalah-rules-v1",
                "source_sha256": None,
            },
            "encoder_identity": {"encoder": "kalah_v3", "source_sha256": None},
            "runtime": runtime_identity(),
            "dependency_versions": {
                "python": None,
                "numpy": None,
                "platform": None,
            },
            "generation": {
                "master_seed": MASTER_SEED,
                "seed_contract": SEED_CONTRACT,
                "opening_reserve": OPENING_RESERVE,
                "invalid_prefix_policy": OPENING_INVALID_PREFIX_POLICY,
                "opening_target": OPENING_TARGET,
                "prefix_actions": PREFIX_ACTIONS,
                "games_per_opening": GAMES_PER_OPENING,
                "continuation_action_limit": MAX_CONTINUATION_ACTIONS,
            },
            "membership_snapshot_sha256": {
                "B512": None,
                "S615": None,
                "protected_holdout": None,
            },
            "membership_snapshot_file_sha256": {
                "B512": None,
                "S615": None,
                "protected_holdout": None,
            },
            "membership_source_manifest_sha256": None,
            "source_provenance_sha256": None,
            "provenance_source_file_sha256": {},
            "selection": {
                "targets": STONE_TARGETS,
                "max_per_game": 2,
                "max_per_opening": 4,
                "minimum_openings": MIN_OPENING_CLUSTERS,
                "objective": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity",
                "optimizer": "seed624-dfs-bounded-v2",
                "candidate_limit": MAX_SELECTION_CANDIDATES,
                "runtime_limit_seconds": MAX_SELECTION_SECONDS,
                "node_limit": MAX_SELECTION_NODES,
                "time_check_interval_nodes": SELECTION_TIME_CHECK_INTERVAL_NODES,
                "input_memory_limit_bytes": MAX_SELECTION_INPUT_BYTES,
                "working_memory_limit_bytes": MAX_SELECTION_WORKING_BYTES,
                "workers": 1,
            },
            "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
            "stone_bin_weights": STONE_BIN_WEIGHTS,
            "output_schema": SCHEMA,
            "create_only_output_paths": [],
            "temporary_path": None,
            "production_resource_limits": {
                "wall_seconds": None,
                "cpu_seconds": None,
                "process_rss_bytes": None,
                "execution_concurrency": None,
                "output_and_temporary_storage_bytes": None,
                "status": "registered_in_lifecycle_attempt_and_monitored_by_operator",
            },
            "outputs": {"schema": SCHEMA},
            "verification_commands": [
                "python -m ml.alphazero_lite.verify_seed624_prospective_benchmark <copied-bundle>/population.json",
                "python -m unittest ml.alphazero_lite.test_seed624_prospective_benchmark",
            ],
            "execution_scope": EXECUTION_SCOPE,
            "authorization_requirements": {
                "contract": "seed636-registration-pr-authorization-v1",
                "repository": "useruby/mancala-ai",
                "project_owner_identity": "useruby",
                "registration_pr_merged_into": "main",
                "execution_scope": EXECUTION_SCOPE,
                "registration_sha256": None,
            },
        },
    }


def main() -> None:
    """Lifecycle-runner entry point; direct runs lack the required started receipt."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate_population({"attempt_dir": str(args.attempt_dir)}, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
