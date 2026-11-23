"""Seed 624 deterministic, model-independent prospective state population.

The module has no model, search, solver, or training imports.  Production
generation is deliberately guarded by an explicit authorization record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
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
# Trust anchors are provisioned only through a reviewed source change. An empty
# production allowlist deliberately prevents approval by caller-supplied keys.
TRUSTED_PROJECT_OWNER_KEYS: dict[str, dict[str, str]] = {}
EXECUTION_SCOPE = "model-independent-generation-and-screening"
FAILURE_CATEGORIES = frozenset(
    {
        "opening_reserve_exhaustion",
        "insufficient_stone_bin_supply",
        "mathematical_infeasibility",
        "selector_resource_exhaustion",
        "verification_or_identity_failure",
        "unexpected_execution_exception",
        "external_resource_limit_termination",
        "preflight_failure",
    }
)
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
    "ml/alphazero_lite/seed629_execution_supervisor.py",
    "ml/alphazero_lite/verify_seed629_supervisor.py",
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
    try:
        crypto_version = importlib.metadata.version("cryptography")
    except importlib.metadata.PackageNotFoundError:
        crypto_version = "unavailable"
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "prng": f"random.Random/MT19937; Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "optimizer": "seed624-dfs-bounded-v2",
        "cryptography": crypto_version,
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


def verify_owner_approval_record(
    record: dict[str, Any], freeze_sha256: str, execution_scope: str
) -> str:
    """Validate Ed25519 approval against repository-pinned trust anchors."""
    if (
        record.get("schema") != "seed624-owner-approval-v1"
        or record.get("freeze_sha256") != freeze_sha256
        or record.get("execution_scope") != execution_scope
        or not isinstance(record.get("project_owner_identity"), str)
    ):
        raise PermissionError("owner_approval_record_binding_mismatch")
    key_record = TRUSTED_PROJECT_OWNER_KEYS.get(str(record.get("key_id")))
    if key_record is None or key_record.get("project_owner_identity") != record.get(
        "project_owner_identity"
    ):
        raise PermissionError("owner_approval_signing_key_not_trusted")
    try:
        from base64 import b64decode
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise PermissionError("owner_approval_crypto_dependency_unavailable") from error
    try:
        signature = b64decode(record["signature_ed25519"], validate=True)
        payload = {
            key: value for key, value in record.items() if key != "signature_ed25519"
        }
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(key_record["public_key_hex"])
        ).verify(signature, canonical_bytes(payload))
    except (KeyError, ValueError, TypeError, InvalidSignature) as error:
        raise PermissionError("owner_approval_signature_invalid") from error
    return str(record["project_owner_identity"])


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
            continue
        canonical_seen[cid] = index
        float_seen[fid] = index
        chosen.append(opening)
        attempts.append(
            {
                "opening_index": index,
                "seed": seed,
                "status": "accepted",
                "canonical_identity": cid,
                "production_float32_identity": fid,
            }
        )
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
    opening: dict[str, Any], master_seed: int, trajectory_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opening_index = int(opening["opening_index"])
    game_id = f"o{opening_index:04d}-g{trajectory_index:02d}"
    game = KalahGame.from_state(opening["state"])
    rng = random.Random(
        derive_seed(master_seed, opening_index, trajectory_index, "continuation")
    )
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
        "seed": derive_seed(
            master_seed, opening_index, trajectory_index, "continuation"
        ),
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


class ExternalResourceLimitError(RuntimeError):
    """Supervisor reported termination for an independently enforced limit."""


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
        screened.append(
            {
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
        )
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
    """Production entry point requiring a repository-trusted owner signature."""
    freeze_path = authorization.get("execution_freeze_path")
    if not freeze_path or not Path(freeze_path).is_file():
        raise PermissionError("execution_freeze_file_required")
    freeze_raw = Path(freeze_path).read_bytes()
    if sha256(freeze_raw) != authorization["execution_freeze_sha256"]:
        raise PermissionError("execution_freeze_hash_mismatch")
    freeze = json.loads(freeze_raw)
    approval_path = authorization.get("owner_approval_record_path")
    if not approval_path or not Path(approval_path).is_file():
        raise PermissionError("signed_owner_approval_record_required")
    approval_raw = Path(approval_path).read_bytes()
    approval_record = json.loads(approval_raw)
    owner_identity = verify_owner_approval_record(
        approval_record,
        authorization["execution_freeze_sha256"],
        EXECUTION_SCOPE,
    )
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
    if (
        freeze.get("schema") != "seed624-execution-freeze-v1"
        or freeze.get("status") != "authorized"
        or freeze.get("project_owner_identity") != owner_identity
        or freeze.get("execution_scope") != EXECUTION_SCOPE
        or authorization.get("authorized_run_identity")
        != freeze.get("authorized_run_identity")
        or authorization.get("failure_receipt_path")
        != freeze.get("failure_receipt_contract", {}).get("create_only_path")
        or freeze.get("failure_receipt_contract", {}).get("schema")
        != "seed624-failure-receipt-contract-v2"
        or freeze.get("failure_receipt_contract", {}).get("authenticated_record_schema")
        != "seed624-authenticated-execution-failure-v1"
        or freeze.get("failure_receipt_contract", {}).get(
            "unauthenticated_preflight_record_schema"
        )
        != "seed624-unauthenticated-preflight-failure-v1"
        or freeze.get("failure_receipt_contract", {}).get("categories")
        != sorted(FAILURE_CATEGORIES)
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
        or not production_resources_valid
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
    opening_selection = choose_distinct_openings(
        lambda index: generate_opening(MASTER_SEED, index),
        reserve=OPENING_RESERVE,
        target=OPENING_TARGET,
    )
    openings = opening_selection["openings"]
    games: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for opening in openings:
        for trajectory_index in range(GAMES_PER_OPENING):
            game, occurrences = generate_game(opening, MASTER_SEED, trajectory_index)
            game["opening_prefix_actions"] = opening["prefix_actions"]
            games.append(game)
            candidates.extend(occurrences)
    unique = deduplicate_candidates(candidates)
    screened = screen_known_exposure(
        unique,
        evidence,
        expected_snapshot_hashes=freeze.get("membership_snapshot_sha256"),
        expected_provenance_hash=freeze.get("source_provenance_sha256"),
    )
    selection = select_cohort(screened)
    bundle = {
        "schema": SCHEMA,
        "authorization": {
            "project_owner_identity": owner_identity,
            "execution_freeze_sha256": authorization["execution_freeze_sha256"],
            "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
            "owner_approval_record": approval_record,
            "owner_approval_record_raw_utf8": approval_raw.decode("utf-8"),
            "owner_approval_record_sha256": sha256(approval_raw),
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
    output.mkdir(parents=True)
    (output / "population.json").write_bytes(canonical_bytes(bundle) + b"\n")
    (output / "population.sha256").write_text(
        sha256((output / "population.json").read_bytes()) + "\n"
    )
    return {
        "schema": SCHEMA,
        "openings": len(openings),
        "games": len(games),
        "eligible_candidates": len(screened),
        "selected": len(selection["states"]),
        "output": str(output),
    }


def _failure_receipt_binding(
    authorization: dict[str, Any], receipt_path: Path | None = None
) -> dict[str, Any]:
    """Authenticate the exact freeze and caller-supplied receipt bindings."""
    freeze_path = authorization.get("execution_freeze_path")
    if not freeze_path or not Path(freeze_path).is_file():
        raise PermissionError("execution_freeze_file_required")
    freeze_raw = Path(freeze_path).read_bytes()
    digest = sha256(freeze_raw)
    if digest != authorization.get("execution_freeze_sha256"):
        raise PermissionError("execution_freeze_hash_mismatch")
    freeze = json.loads(freeze_raw)
    approval_path = authorization.get("owner_approval_record_path")
    if not approval_path or not Path(approval_path).is_file():
        raise PermissionError("signed_owner_approval_record_required")
    approval = json.loads(Path(approval_path).read_bytes())
    owner = verify_owner_approval_record(approval, digest, EXECUTION_SCOPE)
    if (
        freeze.get("schema") != "seed624-execution-freeze-v1"
        or freeze.get("status") != "authorized"
        or freeze.get("execution_scope") != EXECUTION_SCOPE
        or freeze.get("project_owner_identity") != owner
        or freeze.get("failure_receipt_contract", {}).get("schema")
        != "seed624-failure-receipt-contract-v2"
        or freeze.get("failure_receipt_contract", {}).get("authenticated_record_schema")
        != "seed624-authenticated-execution-failure-v1"
        or freeze.get("failure_receipt_contract", {}).get(
            "unauthenticated_preflight_record_schema"
        )
        != "seed624-unauthenticated-preflight-failure-v1"
        or freeze.get("failure_receipt_contract", {}).get("categories")
        != sorted(FAILURE_CATEGORIES)
    ):
        raise PermissionError("failure_receipt_freeze_not_authorized")
    frozen_run_identity = freeze.get("authorized_run_identity")
    frozen_receipt_path = freeze.get("failure_receipt_contract", {}).get(
        "create_only_path"
    )
    if not isinstance(frozen_run_identity, str) or not frozen_run_identity:
        raise PermissionError("failure_receipt_run_identity_missing_from_freeze")
    if not isinstance(frozen_receipt_path, str) or not frozen_receipt_path:
        raise PermissionError("failure_receipt_path_missing_from_freeze")
    if authorization.get("authorized_run_identity") != frozen_run_identity:
        raise PermissionError("failure_receipt_run_identity_mismatch")
    if authorization.get("failure_receipt_path") != frozen_receipt_path:
        raise PermissionError("failure_receipt_path_mismatch")
    if receipt_path is not None and str(receipt_path) != frozen_receipt_path:
        raise PermissionError("failure_receipt_path_mismatch")
    return {
        "freeze_sha256": digest,
        "freeze_raw": freeze_raw,
        "freeze": freeze,
        "approval": approval,
        "owner_identity": owner,
        "authorized_run_identity": frozen_run_identity,
        "failure_receipt_path": Path(frozen_receipt_path),
    }


def _create_durable_json(path: Path, payload: dict[str, Any]) -> Path:
    """Exclusive-create JSON, fsyncing both file contents and directory entry."""
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"failure_receipt_parent_directory_missing:{path.parent}"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Do not remove a possibly complete or partially written forensic record.
        raise
    return path


def write_failure_receipt(
    *,
    authorization: dict[str, Any],
    category: str,
    diagnostic: dict[str, Any],
) -> Path:
    """Write an authenticated receipt whose identity/path come from a signed freeze."""
    if category not in FAILURE_CATEGORIES - {"preflight_failure"}:
        raise ValueError("invalid_authenticated_failure_category")
    binding = _failure_receipt_binding(authorization)
    payload = {
        "schema": "seed624-authenticated-execution-failure-v1",
        "execution_freeze_sha256": binding["freeze_sha256"],
        "authorized_run_identity": binding["authorized_run_identity"],
        "failure_category": category,
        "diagnostic": diagnostic,
    }
    return _create_durable_json(binding["failure_receipt_path"], payload)


def _write_preflight_failure(
    *, authorization: dict[str, Any], requested_path: Path | None, error: BaseException
) -> Path:
    """Record an unauthenticated preflight failure without claiming a run identity."""
    path = requested_path or Path(
        authorization.get("failure_receipt_path", "seed624-preflight-failure.json")
    )
    return _create_durable_json(
        path,
        {
            "schema": "seed624-unauthenticated-preflight-failure-v1",
            "execution_freeze_sha256": None,
            "authorized_run_identity": None,
            "failure_category": "preflight_failure",
            "diagnostic": {
                "exception_type": type(error).__name__,
                "message": str(error),
                "claimed_freeze_sha256": authorization.get("execution_freeze_sha256"),
                "claimed_run_identity": None,
            },
        },
    )


def _failure_category(error: BaseException) -> str:
    if isinstance(error, ExternalResourceLimitError):
        return "external_resource_limit_termination"
    if isinstance(error, ShortfallError):
        reason = str(error.report.get("reason", ""))
        if reason == "opening_reserve_exhausted":
            return "opening_reserve_exhaustion"
        if reason == "hard_supply":
            return "insufficient_stone_bin_supply"
        if reason in {"aggregate_cluster_capacity", "constraints_infeasible"}:
            return "mathematical_infeasibility"
        if "resource_limit" in reason:
            return "selector_resource_exhaustion"
        return "unexpected_execution_exception"
    if isinstance(error, (PermissionError, ValueError, KeyError, FileNotFoundError)):
        return "verification_or_identity_failure"
    return "unexpected_execution_exception"


def execute_with_failure_receipts(
    authorization: dict[str, Any], output: Path, *, receipt_path: Path | None = None
) -> dict[str, Any]:
    """Run after authenticating receipt bindings; preserve both failures if writing fails."""
    try:
        _failure_receipt_binding(authorization, receipt_path)
    except BaseException as preflight_error:
        try:
            _write_preflight_failure(
                authorization=authorization,
                requested_path=receipt_path,
                error=preflight_error,
            )
        except BaseException as receipt_error:
            preflight_error.add_note(
                f"Unauthenticated preflight receipt also failed: "
                f"{type(receipt_error).__name__}: {receipt_error}"
            )
        raise
    try:
        return generate_population(authorization, output)
    except BaseException as error:
        diagnostic: dict[str, Any] = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "partial_output_path": str(output),
            "partial_output_exists": output.exists(),
        }
        if isinstance(error, ShortfallError):
            diagnostic["report"] = error.report
        try:
            write_failure_receipt(
                authorization=authorization,
                category=_failure_category(error),
                diagnostic=diagnostic,
            )
        except BaseException as receipt_error:
            error.add_note(
                f"Authenticated failure receipt also failed: "
                f"{type(receipt_error).__name__}: {receipt_error}"
            )
        raise


def proposed_freeze_schema() -> dict[str, Any]:
    """Return a schema template, never a production-ready execution freeze."""
    return {
        "schema": "seed624-proposed-execution-freeze-v1",
        "status": "proposal_not_ready_or_approved",
        "required": {
            "git_commit": None,
            "source_file_sha256": {name: None for name in SOURCE_FILES},
            "independent_verifier_sha256": None,
            "external_harness_identity": {
                "name": None,
                "version": None,
                "sha256": None,
            },
            "supervisor_contract": {
                "implementation": "seed629-cgroup-v2-v1",
                "launch_status": "disabled_pending_verified_hard_quota_backend",
                "source_file": "ml/alphazero_lite/seed629_execution_supervisor.py",
                "source_sha256": None,
                "evidence_verifier": "ml.alphazero_lite.verify_seed629_supervisor",
                "execution_scope": "dedicated-cgroup-v2-with-hard-filesystem-quota",
                "memory_contract": "memory.max is conservatively below frozen process RSS bytes; cgroup memory accounting is not RSS-equivalent",
                "storage_contract": "hard project/filesystem quota covers output and temporary storage; unsupported quota fails closed",
            },
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
                "cryptography": None,
                "platform": None,
                "kernel": None,
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
            "supervisor_result_path": None,
            "worker_command": None,
            "production_resource_limits": {
                "wall_seconds": None,
                "cpu_seconds": None,
                "process_rss_bytes": None,
                "execution_concurrency": None,
                "output_and_temporary_storage_bytes": None,
                "status": "must_be_set_and_enforced_by_authorized_execution_harness",
            },
            "authorized_run_identity": None,
            "failure_receipt_contract": {
                "schema": "seed624-failure-receipt-contract-v2",
                "authenticated_record_schema": "seed624-authenticated-execution-failure-v1",
                "unauthenticated_preflight_record_schema": "seed624-unauthenticated-preflight-failure-v1",
                "create_only_path": None,
                "categories": [
                    *sorted(FAILURE_CATEGORIES),
                ],
                "preserve_partial_artifacts": True,
                "automatic_retry": False,
            },
            "outputs": {"schema": SCHEMA},
            "verification_commands": [
                "python -m ml.alphazero_lite.verify_seed624_prospective_benchmark <copied-bundle>/population.json",
                "python -m unittest ml.alphazero_lite.test_seed624_prospective_benchmark",
            ],
            "project_owner_identity": None,
            "execution_scope": EXECUTION_SCOPE,
            "owner_approval_record": {
                "schema": "seed624-owner-approval-v1",
                "key_id": None,
                "freeze_sha256": None,
                "execution_scope": EXECUTION_SCOPE,
                "signature_ed25519": None,
            },
            "trusted_key_requirement": "signing key ID must resolve only in TRUSTED_PROJECT_OWNER_KEYS from reviewed repository source; caller-supplied public keys are rejected",
            "approval_requirements": {
                "project_owner_identity": None,
                "explicit_protocol_amendment_approval_record": None,
                "exact_freeze_digest_signature": None,
                "execution_authorization_record": None,
                "key_provisioning_is_authorization": False,
            },
        },
    }
