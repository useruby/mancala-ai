"""Independent Seed 624 verifier for a copied synthetic or authorized bundle."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from sys import getsizeof
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

MASTER_SEED = 62320261121
BIN_TARGETS = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
BIN_WEIGHTS = {str(stones): 0.2 for stones in BIN_TARGETS}
ROOT = Path(__file__).resolve().parents[2]
MAX_SELECTION_CANDIDATES = 10_240
MAX_SELECTION_NODES = 250_000
MAX_SELECTION_SECONDS = 30
SELECTION_TIME_CHECK_INTERVAL_NODES = 1
MAX_SELECTION_WORKING_BYTES = 64 * 1024 * 1024
MAX_SELECTION_INPUT_BYTES = 128 * 1024 * 1024
EXECUTION_SCOPE = "model-independent-generation-and-screening-only"
INVALID_PREFIX_POLICY = "consume-reserve-index-skip-record-provenance-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fetch_github_comment(comment_id: str) -> dict[str, Any]:
    return json.loads(
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/useruby/mancala-ai/issues/comments/{comment_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def _git_blob(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def verify_owner_approval(record: dict[str, Any], lifecycle: dict[str, Any]) -> str:
    """Independently validate recorded GitHub owner-comment evidence."""
    approval_marker = "SEED636-APPROVAL-V1"
    project_owner = "useruby"
    repository = "useruby/mancala-ai"
    approved_decisions = [
        "seed625-finite-cohort-estimand-amendment",
        "seed635-worktree-execution-contract-amendment",
        "known-exposure-screened-inherited-PE4-unresolved",
        "model-independent-population-generation-and-screening-only",
    ]
    required_documents = {
        "docs/seed624-estimand-amendment-proposal.md",
        "docs/seed623-prospective-exact-quality-benchmark.md",
        "docs/seed623-631-execution-contract-amendment.md",
        "docs/experiment-execution-protocol.md",
    }

    registration_raw = lifecycle.get("registration_raw_utf8", "").encode()
    registration = json.loads(registration_raw)
    if _sha(registration_raw) != lifecycle.get("registration_sha256"):
        raise ValueError("approval_registration_hash_mismatch")
    if record.get("schema") != "seed636-github-owner-approval-v1":
        raise ValueError("github_owner_approval_missing_or_invalid")
    if record.get("project_owner_identity") != project_owner:
        raise ValueError("github_project_owner_identity_mismatch")
    if record.get("execution_scope") != EXECUTION_SCOPE:
        raise ValueError("github_approval_execution_scope_mismatch")
    if record.get("registration_sha256") != lifecycle.get("registration_sha256"):
        raise ValueError("github_approval_registration_identity_mismatch")
    if record.get("protocol_documents") != registration.get("protocol_documents"):
        raise ValueError("github_approval_protocol_document_mismatch")
    if (
        registration.get("approved_decisions") != approved_decisions
        or record.get("approved_decisions") != approved_decisions
    ):
        raise ValueError("github_approval_decision_scope_mismatch")
    if record.get("approved_protocol_revision") != registration.get(
        "approved_protocol_revision"
    ):
        raise ValueError("github_approval_protocol_revision_mismatch")
    approved_revision = str(record.get("approved_protocol_revision", ""))
    if approved_revision != registration.get("base_commit"):
        raise ValueError("github_approval_protocol_revision_not_base_commit")
    if not required_documents.issubset(
        record.get("protocol_documents", {})
    ) or not re.fullmatch(r"[0-9a-f]{40,64}", approved_revision):
        raise ValueError("github_approval_protocol_identity_invalid")
    try:
        resolved_revision = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                f"{approved_revision}^{{commit}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("github_approval_protocol_revision_missing") from error
    if resolved_revision != approved_revision:
        raise ValueError("github_approval_protocol_revision_not_canonical")
    for name, expected_digest in record["protocol_documents"].items():
        raw_document = ROOT / name
        if (
            name.startswith("/")
            or ".." in Path(name).parts
            or raw_document.is_symlink()
        ):
            raise ValueError("github_approval_protocol_document_path_invalid")
        document = raw_document.resolve(strict=True)
        try:
            approved_blob = _git_blob(approved_revision, name)
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "github_approval_protocol_document_missing_at_revision"
            ) from error
        if (
            not document.is_file()
            or not document.is_relative_to(ROOT)
            or _sha(approved_blob) != expected_digest
            or document.read_bytes() != approved_blob
        ):
            raise ValueError("github_approval_protocol_document_hash_mismatch")
    comment = record.get("github_comment_evidence", {})
    if (
        comment.get("author") != project_owner
        or comment.get("created_at") != record.get("approval_timestamp")
        or comment.get("html_url") != record.get("approval_url")
        or not str(comment.get("html_url", "")).startswith(
            f"https://github.com/{repository}/"
        )
    ):
        raise ValueError("github_approval_comment_evidence_mismatch")
    match = re.fullmatch(
        r"https://github\.com/useruby/mancala-ai/(?:issues|pull)/(\d+)#issuecomment-(\d+)",
        str(record.get("approval_url", "")),
    )
    if not match:
        raise ValueError("github_approval_comment_url_invalid")
    api_record = _fetch_github_comment(match.group(2))
    if (
        api_record.get("id") != int(match.group(2))
        or comment.get("id") != int(match.group(2))
        or api_record.get("user", {}).get("login") != project_owner
        or api_record.get("created_at") != record.get("approval_timestamp")
        or api_record.get("html_url") != record.get("approval_url")
        or api_record.get("body") != comment.get("body")
    ):
        raise ValueError("github_approval_live_record_mismatch")
    expected = {
        "marker": approval_marker,
        "experiment_id": registration.get("experiment_id"),
        "registration_sha256": lifecycle.get("registration_sha256"),
        "approved_protocol_revision": registration.get("approved_protocol_revision"),
        "protocol_documents": registration.get("protocol_documents"),
        "approved_decisions": approved_decisions,
        "execution_scope": EXECUTION_SCOPE,
    }
    body = str(comment.get("body", ""))
    prefix = approval_marker + "\n"
    if not body.startswith(prefix) or json.loads(body[len(prefix) :]) != expected:
        raise ValueError("github_approval_comment_binding_mismatch")
    return project_owner


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _deep_size_bytes(value: Any) -> int:
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


def _identities(state: dict[str, Any]) -> tuple[str, str]:
    canonical = _sha(_canonical(state))
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return canonical, _sha(vector.tobytes())


def _seed(master: int, opening: int, trajectory: int, phase: str) -> int:
    msg = _canonical(
        {
            "domain": "useruby.mancala-ai/seed624/seed/v1",
            "master_seed": master,
            "opening_index": opening,
            "trajectory_index": trajectory,
            "phase": phase,
        }
    )
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def _rank(row: dict[str, Any]) -> str:
    return _sha(
        _canonical(
            {
                "domain": "useruby.mancala-ai/seed624/candidate-rank/v1",
                "master_seed": MASTER_SEED,
                "opening_id": row["opening_identity"],
                "game_id": row["game_id"],
                "stone_bin": row["active_pit_stones"],
                "canonical_identity": row["canonical_identity"],
            }
        )
    )


def validate_candidate_rank(row: dict[str, Any]) -> None:
    if row.get("rank") != _rank(row):
        raise ValueError("candidate_rank_mismatch")


def _reconstruct_selection(
    candidates: list[dict[str, Any]],
    targets: dict[int, int],
    minimum_openings: int,
) -> tuple[list[str], int, float, int, int]:
    """Verifier-owned bounded rank-first search; does not trust producer output."""
    if len(candidates) > MAX_SELECTION_CANDIDATES:
        raise ValueError("verifier_candidate_count_limit")
    input_bytes = _deep_size_bytes(candidates)
    if input_bytes > MAX_SELECTION_INPUT_BYTES:
        raise ValueError("verifier_input_memory_limit")
    bins = tuple(sorted(targets))
    eligible = [
        row
        for row in candidates
        if row.get("classification")
        == "known_exposure_nonmatch_inherited_PE4_unresolved"
    ]
    grouped = {
        stones: sorted(
            (row for row in eligible if int(row["active_pit_stones"]) == stones),
            key=lambda row: (row["rank"], row["canonical_identity"]),
        )
        for stones in bins
    }
    total = sum(targets.values())
    if (
        sum(
            min(2, count)
            for count in Counter(row["game_id"] for row in eligible).values()
        )
        < total
        or sum(
            min(4, count)
            for count in Counter(int(row["opening_index"]) for row in eligible).values()
        )
        < total
        or len({row["game_id"] for row in eligible}) < (total + 1) // 2
        or len({int(row["opening_index"]) for row in eligible}) < minimum_openings
    ):
        raise ValueError("published_selection_not_feasible")
    chosen: list[dict[str, Any]] = []
    games: Counter[str] = Counter()
    openings: Counter[int] = Counter()
    opening_ids: set[int] = set()
    started = time.monotonic()
    node_count = 0
    tracing_was_active = tracemalloc.is_tracing()
    if not tracing_was_active:
        tracemalloc.start()

    def visit(bin_index: int, offset: int) -> bool:
        nonlocal node_count
        node_count += 1
        if node_count > MAX_SELECTION_NODES:
            raise ValueError("verifier_optimizer_node_limit")
        if (
            node_count % SELECTION_TIME_CHECK_INTERVAL_NODES == 0
            and time.monotonic() - started > MAX_SELECTION_SECONDS
        ):
            raise ValueError("verifier_optimizer_resource_limit")
        if node_count % 128 == 0:
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            if peak_bytes > MAX_SELECTION_WORKING_BYTES:
                raise ValueError("verifier_optimizer_memory_limit")
        if bin_index == len(bins):
            return len(opening_ids) >= minimum_openings
        stones = bins[bin_index]
        quota = targets[stones]
        have = sum(int(item["active_pit_stones"]) == stones for item in chosen)
        need = quota - have
        if need <= 0:
            return visit(bin_index + 1, 0)
        pool = grouped[stones]
        possible_count = sum(
            1
            for index in range(offset, len(pool))
            if games[pool[index]["game_id"]] < 2
            and openings[int(pool[index]["opening_index"])] < 4
        )
        if possible_count < need:
            return False
        skipped_count = 0
        for index in range(offset, len(pool)):
            row = pool[index]
            if games[row["game_id"]] >= 2 or openings[int(row["opening_index"])] >= 4:
                continue
            opening = int(row["opening_index"])
            chosen.append(row)
            games[row["game_id"]] += 1
            openings[opening] += 1
            is_new = opening not in opening_ids
            opening_ids.add(opening)
            if visit(bin_index, index + 1):
                return True
            if is_new and openings[opening] == 1:
                opening_ids.remove(opening)
            openings[opening] -= 1
            games[row["game_id"]] -= 1
            chosen.pop()
            skipped_count += 1
            if possible_count - skipped_count < need:
                break
        return False

    try:
        feasible = visit(0, 0)
    finally:
        elapsed = time.monotonic() - started
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        if not tracing_was_active:
            tracemalloc.stop()
    if not feasible:
        raise ValueError("published_selection_not_feasible")
    return (
        [
            row["canonical_identity"]
            for row in sorted(
                chosen,
                key=lambda row: (
                    int(row["active_pit_stones"]),
                    row["rank"],
                    row["canonical_identity"],
                ),
            )
        ],
        node_count,
        elapsed,
        peak_bytes,
        input_bytes,
    )


def _opening_state() -> KalahGame:
    return KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0)


def _replay_opening_seed(index: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    game = _opening_state()
    seed = _seed(MASTER_SEED, index, 0, "opening")
    rng = random.Random(seed)
    moves = []
    for _ in range(8):
        legal = game.possible_moves()
        if not legal:
            terminal = game.over()
            state = game.to_state()
            cid, fid = _identities(state)
            reason = (
                "terminal_before_eight_actions"
                if terminal
                else "no_legal_action_before_prefix_complete"
            )
            return (
                {
                    "opening_index": index,
                    "seed": seed,
                    "status": "rejected_invalid_prefix",
                    "reason": reason,
                    "actions": moves,
                    "terminal": terminal,
                    "state": state,
                    "canonical_identity": cid,
                    "production_float32_identity": fid,
                },
                None,
            )
        action = rng.choice(legal)
        if not game.move(game.pit_index(action)):
            raise ValueError("opening_rng_replay_failed")
        moves.append(action)
        terminal = game.over()
        if terminal:
            state = game.to_state()
            cid, fid = _identities(state)
            reason = (
                "terminal_at_eight_actions"
                if len(moves) == 8
                else "terminal_before_eight_actions"
            )
            return (
                {
                    "opening_index": index,
                    "seed": seed,
                    "status": "rejected_invalid_prefix",
                    "reason": reason,
                    "actions": moves,
                    "terminal": True,
                    "state": state,
                    "canonical_identity": cid,
                    "production_float32_identity": fid,
                },
                None,
            )
    cid, fid = _identities(game.to_state())
    opening = {
        "opening_index": index,
        "seed": seed,
        "state": game.to_state(),
        "canonical_identity": cid,
        "production_float32_identity": fid,
        "prefix_actions": moves,
    }
    return (
        {
            "opening_index": index,
            "seed": seed,
            "status": "valid_prefix",
            "canonical_identity": cid,
            "production_float32_identity": fid,
        },
        opening,
    )


def _reconstruct_opening_reserve(
    consumed: list[int], target: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    openings: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    canonical_seen: dict[str, int] = {}
    float_seen: dict[str, int] = {}
    for raw_index in consumed:
        index = int(raw_index)
        attempt, opening = _replay_opening_seed(index)
        if opening is None:
            attempts.append(attempt)
            rejections.append(attempt)
            continue
        cid = opening["canonical_identity"]
        fid = opening["production_float32_identity"]
        if cid in canonical_seen:
            attempt = {
                "opening_index": index,
                "seed": opening["seed"],
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
                "seed": opening["seed"],
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
        openings.append(opening)
        attempts.append(
            {
                "opening_index": index,
                "seed": opening["seed"],
                "status": "accepted",
                "canonical_identity": cid,
                "production_float32_identity": fid,
            }
        )
        if len(openings) == target:
            if index != consumed[-1]:
                raise ValueError("opening_reserve_consumed_after_target_reached")
            break
    if len(openings) != target:
        raise ValueError("opening_reserve_distinct_valid_shortfall")
    return openings, attempts, rejections


def _apply_actions(game: KalahGame, actions: list[int], *, label: str) -> None:
    for ply, action in enumerate(actions):
        if game.over():
            raise ValueError(f"terminal_transition:{label}:{ply}")
        if action not in game.possible_moves() or not game.move(game.pit_index(action)):
            raise ValueError(f"illegal_trajectory_action:{label}:{ply}")


def _expected_screened(
    candidates: list[dict],
    evidence: dict,
    *,
    expected_hashes: dict[str, str],
    expected_provenance_hash: str | None,
) -> list[dict]:
    names = ("B512", "S615", "protected_holdout")
    inventory: dict[str, tuple[set[str], set[str]]] = {}
    for name in names:
        source = evidence.get(name)
        if not isinstance(source, dict):
            raise ValueError(f"unauthenticated_membership:{name}")
        rows = source.get("rows")
        if (
            not isinstance(rows, list)
            or not rows
            or _sha(_canonical(rows)) != source.get("sha256")
            or expected_hashes.get(name) != _sha(_canonical(rows))
        ):
            raise ValueError(f"membership_snapshot_invalid:{name}")
        ids, floats = set(), set()
        for row in rows:
            cid, fid = _identities(row["canonical_state"])
            if cid != row.get("canonical_identity") or fid != row.get(
                "production_float32_identity"
            ):
                raise ValueError(f"membership_identity_invalid:{name}")
            ids.add(cid)
            floats.add(fid)
        inventory[name] = ids, floats
    prov = evidence.get("source_provenance", {})
    prov_authenticated = expected_provenance_hash is not None
    if prov_authenticated:
        if not prov:
            raise ValueError("source_provenance_missing")
        if prov.get("authenticated") is not True:
            raise ValueError("source_provenance_unauthenticated")
        prov_payload = {
            key: value
            for key, value in prov.items()
            if key not in {"authenticated", "sha256"}
        }
        prov_hash = _sha(_canonical(prov_payload))
        if prov.get("sha256") != prov_hash or expected_provenance_hash != prov_hash:
            raise ValueError("source_provenance_hash_mismatch")
    elif prov.get("authenticated") is True:
        raise ValueError("source_provenance_hash_required")
    prov_crosswalkable = (
        prov_authenticated and prov.get("crosswalk_status") == "complete"
    )
    if prov_authenticated and prov.get("crosswalk_status") not in {
        "complete",
        "authenticated_non_crosswalkable",
    }:
        raise ValueError("source_provenance_crosswalk_status_invalid")
    if (
        prov.get("crosswalk_status") == "authenticated_non_crosswalkable"
        and prov.get("crosswalk_rows") != []
    ):
        raise ValueError("source_provenance_crosswalk_claim_invalid")
    if prov_crosswalkable and any(
        not isinstance(prov.get(name), list) for name in names
    ):
        raise ValueError("source_provenance_incomplete")
    result = []
    for row in candidates:
        cid, fid = row["canonical_identity"], row["production_float32_identity"]
        hits = {
            name: cid in ids or fid in floats
            for name, (ids, floats) in inventory.items()
        }
        linked: bool | None = False if prov_crosswalkable else None
        if prov_crosswalkable:
            linked = any(
                entry.get("opening_identity") == row["opening_identity"]
                or entry.get("game_id") == row["game_id"]
                for name in names
                for entry in prov[name]
            )
        status = (
            "confirmed_known_exposure"
            if any(hits.values()) or linked
            else "known_exposure_nonmatch_inherited_PE4_unresolved"
            if prov_authenticated
            else "invalid_or_insufficiently_authenticated_evidence"
        )
        if (
            row.get("exposure") != hits
            or row.get("source_provenance_match") != linked
            or row.get("source_provenance_status")
            != prov.get("crosswalk_status", "missing_or_unauthenticated")
            or row.get("classification") != status
        ):
            raise ValueError("exposure_classification_mismatch")
        if row.get("inherited_PE4_exposure") != "unresolved":
            raise ValueError("false_training_disjointness_claim")
        result.append(row)
    return result


def verify_bundle(bundle_path: Path) -> dict[str, Any]:
    """Reconstruct opening/game replay and authenticate all published aggregates."""
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    if bundle.get("schema") != "seed624-prospective-population-v1":
        raise ValueError("population_schema_mismatch")
    forbidden_claims = {
        "training_disjoint",
        "training_disjointness",
        "unseen",
        "contamination_free",
    }
    if any(
        key in forbidden_claims and value not in (False, None, "unresolved")
        for key, value in bundle.items()
    ):
        raise ValueError("false_training_disjointness_claim")
    authorization = bundle.get("authorization", {})
    freeze_raw = authorization.get("execution_freeze_raw_utf8", "").encode("utf-8")
    if _sha(freeze_raw) != authorization.get("execution_freeze_sha256"):
        raise ValueError("execution_freeze_hash_mismatch")
    freeze = json.loads(freeze_raw)
    approval_record = authorization.get("owner_approval_record", {})
    lifecycle_auth = authorization.get("lifecycle", {})
    owner_identity = verify_owner_approval(approval_record, lifecycle_auth)
    attempt_raw = lifecycle_auth.get("attempt_manifest_raw_utf8", "").encode()
    if _sha(attempt_raw) != lifecycle_auth.get("attempt_manifest_sha256"):
        raise ValueError("lifecycle_attempt_manifest_hash_mismatch")
    attempt_manifest = json.loads(attempt_raw)
    registration = json.loads(lifecycle_auth["registration_raw_utf8"])
    if (
        attempt_manifest.get("schema") != "mancala-experiment-attempt-v1"
        or attempt_manifest.get("status") != "frozen"
        or attempt_manifest.get("experiment_id") != registration.get("experiment_id")
        or attempt_manifest.get("attempt_id") != lifecycle_auth.get("attempt_id")
        or attempt_manifest.get("worker_count") != 1
        or attempt_manifest.get("source_commit")
        != bundle.get("source_identity", {}).get("git_commit")
        or attempt_manifest.get("configuration", {}).get("execution_scope")
        != EXECUTION_SCOPE
        or attempt_manifest.get("configuration", {}).get("registration_sha256")
        != lifecycle_auth.get("registration_sha256")
        or attempt_manifest.get("configuration", {}).get("seed624_freeze") != freeze
        or not any(
            "seed624_prospective_benchmark" in item
            for item in attempt_manifest.get("command", [])
        )
    ):
        raise ValueError("lifecycle_attempt_manifest_binding_mismatch")
    registered_inputs = {
        str(Path(path).absolute()): digest
        for path, digest in registration.get("input_artifacts", {}).items()
    }
    if registered_inputs != attempt_manifest.get("input_artifacts"):
        raise ValueError("lifecycle_attempt_input_binding_mismatch")
    registered_resources = freeze.get("production_resource_limits", {})
    attempt_resources = attempt_manifest.get("resource_budget", {})
    additional_resources = attempt_resources.get("additional_limits", {})
    if (
        registered_resources.get("wall_seconds")
        != attempt_resources.get("wall_time_limit_seconds")
        or registered_resources.get("cpu_seconds")
        != additional_resources.get("cpu_seconds")
        or registered_resources.get("process_rss_bytes")
        != additional_resources.get("process_rss_bytes")
        or registered_resources.get("execution_concurrency")
        != attempt_manifest.get("worker_count")
        or registered_resources.get("output_and_temporary_storage_bytes")
        != additional_resources.get("output_and_temporary_storage_bytes")
    ):
        raise ValueError("lifecycle_attempt_resource_binding_mismatch")
    if (
        freeze.get("schema") != "seed624-execution-freeze-v1"
        or freeze.get("status") != "frozen"
        or freeze.get("execution_scope") != EXECUTION_SCOPE
        or authorization.get("project_owner_identity") != owner_identity
        or lifecycle_auth.get("experiment_id") != "seed636-prospective-population"
        or not lifecycle_auth.get("attempt_manifest_sha256")
    ):
        raise ValueError("execution_freeze_binding_mismatch")
    production_resources = freeze.get("production_resource_limits")
    if not isinstance(production_resources, dict) or any(
        type(production_resources.get(name)) is not int
        or production_resources[name] <= 0
        for name in (
            "wall_seconds",
            "cpu_seconds",
            "process_rss_bytes",
            "execution_concurrency",
            "output_and_temporary_storage_bytes",
        )
    ):
        raise ValueError("production_resource_freeze_missing_or_invalid")
    frozen_generation = freeze.get("generation", {})
    if (
        frozen_generation.get("master_seed") != MASTER_SEED
        or frozen_generation.get("opening_reserve") != 512
        or frozen_generation.get("invalid_prefix_policy") != INVALID_PREFIX_POLICY
        or frozen_generation.get("prefix_actions") != 8
        or frozen_generation.get("games_per_opening") != 8
        or frozen_generation.get("continuation_action_limit") != 200
        or frozen_generation.get("seed_contract")
        != "seed624-sha256-v1; big-endian u64 prefix; Python random.Random"
    ):
        raise ValueError("generation_freeze_contract_mismatch")
    source_identity = bundle.get("source_identity", {})
    if (
        source_identity.get("git_commit") != freeze.get("git_commit")
        or source_identity.get("runtime") != freeze.get("runtime")
        or source_identity.get("runtime", {}).get("prng")
        != "random.Random/MT19937; Python "
        + ".".join(
            source_identity.get("runtime", {})
            .get("python", "")
            .split(" ")[0]
            .split(".")[:3]
        )
    ):
        raise ValueError("source_runtime_identity_mismatch")
    frozen_sources = freeze.get("source_file_sha256", {})
    source_snapshots = source_identity.get("files", {})
    if set(frozen_sources) != set(source_snapshots):
        raise ValueError("source_file_identity_set_mismatch")
    for name, expected_hash in frozen_sources.items():
        snapshot = source_snapshots[name]
        if (
            snapshot.get("sha256") != expected_hash
            or _sha(snapshot.get("source", "").encode("utf-8")) != expected_hash
            or not (ROOT / name).is_file()
            or _sha((ROOT / name).read_bytes()) != expected_hash
        ):
            raise ValueError(f"source_file_hash_mismatch:{name}")
    if (
        freeze.get("rules_identity")
        != {
            "rule_id": "historical-kalah-rules",
            "rule_version": "repository-kalah-rules-v1",
            "source_sha256": frozen_sources.get("ml/alphazero_lite/kalah_rules.py"),
        }
        or freeze.get("encoder_identity")
        != {
            "encoder": "kalah_v3",
            "source_sha256": frozen_sources.get("ml/alphazero_lite/input_encodings.py"),
        }
        or freeze.get("output_schema") != "seed624-prospective-population-v1"
        or freeze.get("sampling_contract")
        != "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim"
        or freeze.get("stone_bin_weights") != BIN_WEIGHTS
    ):
        raise ValueError("freeze_rules_encoder_or_estimand_mismatch")
    openings = bundle["openings"]
    generation_contract = freeze.get("generation", {})
    if len(openings) != int(generation_contract.get("opening_target", -1)):
        raise ValueError("opening_target_count_mismatch")
    consumed = bundle["generation"].get("opening_seeds_consumed", [])
    expected_consumed = list(
        range(max((int(row["opening_index"]) for row in openings), default=-1) + 1)
    )
    if consumed != expected_consumed or len(consumed) > int(
        generation_contract.get("opening_reserve", -1)
    ):
        raise ValueError("opening_reserve_consumption_mismatch")
    target_openings = int(generation_contract["opening_target"])
    if target_openings <= 0 or target_openings > 256:
        raise ValueError("invalid_frozen_opening_target")
    expected_first_openings, expected_opening_attempts, expected_opening_rejections = (
        _reconstruct_opening_reserve(consumed, target_openings)
    )
    if _canonical(expected_first_openings) != _canonical(openings):
        raise ValueError("opening_reserve_skipped_valid_opening")
    if (
        bundle["generation"].get("invalid_prefix_policy") != INVALID_PREFIX_POLICY
        or _canonical(bundle["generation"].get("opening_attempts"))
        != _canonical(expected_opening_attempts)
        or _canonical(bundle["generation"].get("opening_rejections"))
        != _canonical(expected_opening_rejections)
    ):
        raise ValueError("opening_attempt_provenance_mismatch")
    if bundle.get("opening_attempts_sha256") != _sha(
        _canonical(expected_opening_attempts)
    ):
        raise ValueError("opening_attempt_inventory_hash_mismatch")
    opening_by_index: dict[int, dict] = {}
    canonical_openings: set[str] = set()
    float_openings: set[str] = set()
    if bundle["generation"].get("master_seed") != MASTER_SEED:
        raise ValueError("master_seed_mismatch")
    for opening in openings:
        index = int(opening["opening_index"])
        if index in opening_by_index or not 0 <= index < 512:
            raise ValueError("duplicate_or_out_of_reserve_opening")
        game = _opening_state()
        _apply_actions(game, opening["prefix_actions"], label=f"opening-{index}")
        if len(opening["prefix_actions"]) != 8 or game.over():
            raise ValueError("opening_prefix_contract_mismatch")
        expected_seed = _seed(MASTER_SEED, index, 0, "opening")
        if opening.get("seed") != expected_seed:
            raise ValueError("opening_seed_record_mismatch")
        rng = random.Random(expected_seed)
        expected_moves = []
        rng_game = _opening_state()
        for _ in range(8):
            action = rng.choice(rng_game.possible_moves())
            expected_moves.append(action)
            if not rng_game.move(rng_game.pit_index(action)):
                raise ValueError("opening_rng_replay_failed")
        if expected_moves != opening["prefix_actions"]:
            raise ValueError("opening_seed_derivation_mismatch")
        cid, fid = _identities(game.to_state())
        if (cid, fid) != (
            opening["canonical_identity"],
            opening["production_float32_identity"],
        ):
            raise ValueError("opening_state_identity_mismatch")
        if cid in canonical_openings or fid in float_openings:
            raise ValueError("duplicate_opening_position")
        canonical_openings.add(cid)
        float_openings.add(fid)
        opening_by_index[index] = opening

    if bundle["opening_inventory_sha256"] != _sha(_canonical(openings)):
        raise ValueError("opening_inventory_hash_mismatch")
    games = bundle["games"]
    candidate_rows = bundle["candidates"]
    if bundle["game_inventory_sha256"] != _sha(_canonical(games)):
        raise ValueError("game_inventory_hash_mismatch")
    if len(games) != len(openings) * int(frozen_generation["games_per_opening"]):
        raise ValueError("continuation_game_count_mismatch")
    expected_game_ids = {
        f"o{int(opening['opening_index']):04d}-g{trajectory:02d}"
        for opening in openings
        for trajectory in range(int(frozen_generation["games_per_opening"]))
    }
    if {row["game_id"] for row in games} != expected_game_ids:
        raise ValueError("continuation_game_set_mismatch")
    if bundle["candidate_inventory_sha256"] != _sha(_canonical(candidate_rows)):
        raise ValueError("candidate_inventory_hash_mismatch")
    aggregate = bundle.get("aggregate_counts", {})
    classifications = Counter(row.get("classification") for row in candidate_rows)
    expected_aggregate = {
        "openings": len(openings),
        "opening_seed_indexes_consumed": len(consumed),
        "invalid_opening_prefixes": sum(
            row["status"] == "rejected_invalid_prefix"
            for row in expected_opening_rejections
        ),
        "duplicate_openings": sum(
            row["status"] == "rejected_duplicate" for row in expected_opening_rejections
        ),
        "games": len(games),
        "candidates": len(candidate_rows),
        "confirmed_known_exposure": classifications["confirmed_known_exposure"],
        "known_exposure_nonmatch_inherited_PE4_unresolved": classifications[
            "known_exposure_nonmatch_inherited_PE4_unresolved"
        ],
        "invalid_or_insufficiently_authenticated_evidence": classifications[
            "invalid_or_insufficiently_authenticated_evidence"
        ],
    }
    if aggregate != expected_aggregate:
        raise ValueError("aggregate_count_mismatch")
    game_map: dict[str, dict] = {}
    visit_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for game_row in games:
        game_id = game_row["game_id"]
        if game_id in game_map:
            raise ValueError("duplicate_game_id")
        game_map[game_id] = game_row
        opening = opening_by_index[int(game_row["opening_index"])]
        if (
            game_row["opening_identity"] != opening["canonical_identity"]
            or game_row.get("opening_prefix_actions") != opening["prefix_actions"]
        ):
            raise ValueError("source_opening_link_mismatch")
        trajectory_index = int(game_id.rsplit("-g", 1)[1])
        seed = _seed(
            MASTER_SEED,
            int(game_row["opening_index"]),
            trajectory_index,
            "continuation",
        )
        if seed != game_row["seed"]:
            raise ValueError("continuation_seed_mismatch")
        rng = random.Random(seed)
        game = KalahGame.from_state(opening["state"])
        actions = game_row["actions"]
        if game_row.get("full_actions") != opening["prefix_actions"] + actions:
            raise ValueError("full_trajectory_mismatch")
        for ply in range(min(len(actions), 200)):
            if game.over():
                raise ValueError("trajectory_continues_after_terminal")
            stones = sum(game.pits)
            legal = game.possible_moves()
            if stones in BIN_TARGETS and len(legal) >= 2:
                state = game.to_state()
                cid, fid = _identities(state)
                visit_row = {
                    "state": state,
                    "canonical_identity": cid,
                    "production_float32_identity": fid,
                    "active_pit_stones": stones,
                    "legal_actions": legal,
                    "side_to_move": game.current_player,
                    "opening_index": int(game_row["opening_index"]),
                    "opening_identity": opening["canonical_identity"],
                    "game_id": game_id,
                    "continuation_ply": ply,
                    "trajectory": actions[:ply],
                }
                visit_row["rank"] = _rank(
                    {
                        "opening_identity": visit_row["opening_identity"],
                        "game_id": game_id,
                        "active_pit_stones": stones,
                        "canonical_identity": cid,
                    }
                )
                visit_groups.setdefault((game_id, stones), []).append(visit_row)
            expected_action = rng.choice(legal)
            if expected_action != actions[ply]:
                raise ValueError("continuation_random_stream_mismatch")
            _apply_actions(game, [actions[ply]], label=game_id)
        if len(actions) > 200 or game.to_state() != game_row["final_state"]:
            raise ValueError("continuation_limit_or_final_state_mismatch")
        terminal = game.over()
        if not terminal and len(actions) < 200:
            raise ValueError("nonterminal_continuation_truncated_early")
        if game_row["terminal"] != terminal or game_row["action_limit_reached"] != (
            len(actions) == 200 and not terminal
        ):
            raise ValueError("game_termination_metadata_mismatch")

    reconstructed: dict[tuple[str, int], tuple[dict[str, Any], int]] = {}
    for key, visits in visit_groups.items():
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
        reconstructed[key] = (
            selected_visit,
            int(selected_visit["continuation_ply"]),
        )

    # Every published state must be reachable at its declared exact visit.
    occurrence_rows = []
    for (_game_id, _stones), (source, _ply) in reconstructed.items():
        row = dict(source)
        occurrence_rows.append(row)
    occurrence_rows.sort(key=lambda row: (row["rank"], row["canonical_identity"]))
    expected_candidates: list[dict[str, Any]] = []
    by_canonical: dict[str, dict[str, Any]] = {}
    by_float: dict[str, dict[str, Any]] = {}
    for occurrence in occurrence_rows:
        cid = occurrence["canonical_identity"]
        fid = occurrence["production_float32_identity"]
        link = {
            "opening_index": occurrence["opening_index"],
            "game_id": occurrence["game_id"],
            "continuation_ply": occurrence["continuation_ply"],
            "rank": occurrence["rank"],
        }
        if cid in by_canonical:
            by_canonical[cid]["occurrences"].append(link)
        elif fid in by_float:
            by_float[fid].setdefault(
                "float32_identity_collision_occurrences", []
            ).append({"canonical_identity": cid, **link})
        else:
            occurrence["occurrences"] = [link]
            expected_candidates.append(occurrence)
            by_canonical[cid] = occurrence
            by_float[fid] = occurrence
    candidate_publication_fields = {
        "exposure",
        "source_provenance_match",
        "classification",
        "membership_evidence_sha256",
        "inherited_PE4_exposure",
        "source_provenance_status",
    }
    if len(expected_candidates) != len(candidate_rows):
        raise ValueError("candidate_inventory_omitted_or_added_rows")
    for expected, published in zip(expected_candidates, candidate_rows, strict=True):
        published_base = {
            key: value
            for key, value in published.items()
            if key not in candidate_publication_fields
        }
        if _canonical(expected) != _canonical(published_base):
            raise ValueError("candidate_occurrence_or_deduplication_mismatch")

    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    for row in candidate_rows:
        if any(
            key in forbidden_claims and value not in (False, None, "unresolved")
            for key, value in row.items()
        ):
            raise ValueError("false_training_disjointness_claim")
        cid, fid = _identities(row["state"])
        if (cid, fid) != (
            row["canonical_identity"],
            row["production_float32_identity"],
        ):
            raise ValueError("candidate_identity_mismatch")
        validate_candidate_rank(row)
        if cid in canonical_seen or fid in float_seen:
            raise ValueError("global_candidate_duplicate")
        canonical_seen.add(cid)
        float_seen.add(fid)
        stones = sum(row["state"]["player_pits"]) + sum(row["state"]["opponent_pits"])
        if stones != row["active_pit_stones"] or stones not in BIN_TARGETS:
            raise ValueError("candidate_stone_bin_mismatch")
        game = KalahGame.from_state(row["state"])
        if (
            game.over()
            or game.possible_moves() != row["legal_actions"]
            or len(row["legal_actions"]) < 2
        ):
            raise ValueError("candidate_eligibility_mismatch")
        if (row["game_id"], stones) not in reconstructed:
            raise ValueError("candidate_source_visit_missing")
        source, ply = reconstructed[(row["game_id"], stones)]
        if (
            source["canonical_identity"] != cid
            or source["state"] != row["state"]
            or source["legal_actions"] != row["legal_actions"]
            or source["side_to_move"] != row["side_to_move"]
            or source["opening_index"] != row["opening_index"]
            or source["opening_identity"] != row["opening_identity"]
            or source["trajectory"] != row["trajectory"]
            or source["within_game_bin_visits"] != row["within_game_bin_visits"]
            or source["rank"] != row["rank"]
            or ply != row["continuation_ply"]
        ):
            raise ValueError("candidate_source_game_replay_mismatch")

    expected_hashes = freeze.get("membership_snapshot_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("freeze_membership_snapshot_hashes_missing")
    selected = bundle["selection"]
    if (
        selected.get("sampling_contract")
        != (
            "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim"
        )
        or selected.get("stone_bin_weights") != BIN_WEIGHTS
    ):
        raise ValueError("invalid_sampling_weight_or_estimand_claim")
    if any(
        key
        in {
            "training_disjoint",
            "training_disjointness",
            "unseen",
            "contamination_free",
        }
        and value not in (False, None, "unresolved")
        for key, value in selected.items()
    ):
        raise ValueError("false_training_disjointness_claim")
    from ml.alphazero_lite.seed628_authenticated_screening import (
        PINNED_FILES as PINNED_SEED627_FILES,
    )

    if freeze.get("membership_snapshot_file_sha256") != {
        name.removesuffix(".json"): digest
        for name, digest in PINNED_SEED627_FILES.items()
        if name.endswith(".json") and name != "source-identity-lineage-manifest.json"
    }:
        raise ValueError("freeze_seed627_snapshot_file_hash_mismatch")
    if (
        freeze.get("membership_source_manifest_sha256")
        != PINNED_SEED627_FILES["source-identity-lineage-manifest.json"]
    ):
        raise ValueError("freeze_seed627_manifest_hash_mismatch")
    from ml.alphazero_lite.verify_seed628_authenticated_screening import (
        reconstruct_evidence as reconstruct_authenticated_membership_evidence,
    )

    authenticated_evidence = reconstruct_authenticated_membership_evidence()
    if _canonical(bundle["evidence"]) != _canonical(authenticated_evidence):
        raise ValueError("production_evidence_does_not_match_pinned_seed627_chain")
    _expected_screened(
        candidate_rows,
        bundle["evidence"],
        expected_hashes=expected_hashes,
        expected_provenance_hash=freeze.get("source_provenance_sha256"),
    )
    if (
        selected.get("sampling_contract")
        != (
            "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim"
        )
        or selected.get("stone_bin_weights") != BIN_WEIGHTS
    ):
        raise ValueError("invalid_sampling_weight_or_estimand_claim")
    if any(
        key
        in {
            "training_disjoint",
            "training_disjointness",
            "unseen",
            "contamination_free",
        }
        and value not in (False, None, "unresolved")
        for key, value in selected.items()
    ):
        raise ValueError("false_training_disjointness_claim")
    chosen = selected["states"]
    if selected.get("selected_inventory_sha256") != _sha(_canonical(chosen)):
        raise ValueError("selected_inventory_hash_mismatch")
    targets = {
        int(key): int(value)
        for key, value in selected.get("targets", BIN_TARGETS).items()
    }
    frozen_selection = freeze.get("selection", {})
    if frozen_selection:
        normalized_frozen_targets = {
            int(key): int(value)
            for key, value in frozen_selection.get("targets", {}).items()
        }
        if (
            targets != normalized_frozen_targets
            or frozen_selection.get("max_per_game") != 2
            or frozen_selection.get("max_per_opening") != 4
            or frozen_selection.get("minimum_openings")
            != int(selected.get("minimum_openings", 128))
            or frozen_selection.get("objective")
            != "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity"
            or frozen_selection.get("optimizer") != "seed624-dfs-bounded-v2"
            or frozen_selection.get("workers") != 1
            or frozen_selection.get("candidate_limit") != MAX_SELECTION_CANDIDATES
            or frozen_selection.get("input_memory_limit_bytes")
            != MAX_SELECTION_INPUT_BYTES
            or frozen_selection.get("runtime_limit_seconds") != MAX_SELECTION_SECONDS
            or frozen_selection.get("node_limit") != MAX_SELECTION_NODES
            or frozen_selection.get("time_check_interval_nodes")
            != SELECTION_TIME_CHECK_INTERVAL_NODES
            or frozen_selection.get("working_memory_limit_bytes")
            != MAX_SELECTION_WORKING_BYTES
        ):
            raise ValueError("selection_freeze_contract_mismatch")
    else:
        raise ValueError("selection_freeze_contract_missing")
    supply = Counter(
        int(row["active_pit_stones"])
        for row in candidate_rows
        if row["classification"] == "known_exposure_nonmatch_inherited_PE4_unresolved"
    )
    hard_short = {
        stones: supply[stones] < 2 * target for stones, target in targets.items()
    }
    if any(hard_short.values()):
        raise ValueError("selected_inventory_below_hard_supply_gate")
    expected_counts = Counter(int(row["active_pit_stones"]) for row in chosen)
    if expected_counts != Counter(targets):
        raise ValueError("selected_stone_bin_quota_mismatch")
    eligible_supply = Counter(
        int(row["active_pit_stones"])
        for row in candidate_rows
        if row["classification"] == "known_exposure_nonmatch_inherited_PE4_unresolved"
    )
    if selected.get("eligible_supply") != {
        str(stones): eligible_supply[stones] for stones in targets
    }:
        raise ValueError("eligible_supply_count_mismatch")
    by_game = Counter(row["game_id"] for row in chosen)
    by_opening = Counter(int(row["opening_index"]) for row in chosen)
    if max(by_game.values(), default=0) > 2 or max(by_opening.values(), default=0) > 4:
        raise ValueError("selection_cluster_cap_exceeded")
    if len(by_opening) < int(selected.get("minimum_openings", 128)) or selected[
        "distinct_openings"
    ] != len(by_opening):
        raise ValueError("selection_opening_diversity_failure")
    if (
        selected.get("counts_by_stone")
        != {str(stones): expected_counts[stones] for stones in targets}
        or selected.get("max_per_opening") != max(by_opening.values(), default=0)
        or selected.get("max_per_game") != max(by_game.values(), default=0)
    ):
        raise ValueError("selection_aggregate_mismatch")
    if any(
        row["classification"] != "known_exposure_nonmatch_inherited_PE4_unresolved"
        for row in chosen
    ):
        raise ValueError("selected_exposed_or_unresolved_provenance_row")
    ids = {row["canonical_identity"] for row in candidate_rows}
    candidate_by_id = {row["canonical_identity"]: row for row in candidate_rows}
    if any(row["canonical_identity"] not in ids for row in chosen):
        raise ValueError("selection_contains_unknown_candidate")
    if any(
        _canonical(row) != _canonical(candidate_by_id[row["canonical_identity"]])
        for row in chosen
    ):
        raise ValueError("selected_candidate_provenance_mismatch")
    actual_ids = [row["canonical_identity"] for row in chosen]
    (
        expected_ids,
        verifier_nodes,
        verifier_elapsed,
        verifier_peak,
        verifier_input_bytes,
    ) = _reconstruct_selection(
        candidate_rows,
        targets,
        int(selected.get("minimum_openings", 128)),
    )
    if actual_ids != expected_ids:
        raise ValueError("selection_rank_or_tie_outcome_mismatch")
    optimizer = selected.get("optimizer", {})
    if (
        optimizer.get("name") != "seed624-dfs-bounded-v2"
        or optimizer.get("version") != 2
        or optimizer.get("workers") != 1
        or optimizer.get("candidate_limit") != MAX_SELECTION_CANDIDATES
        or optimizer.get("input_memory_limit_bytes") != MAX_SELECTION_INPUT_BYTES
        or int(optimizer.get("input_bytes", MAX_SELECTION_INPUT_BYTES + 1))
        > MAX_SELECTION_INPUT_BYTES
        or optimizer.get("node_limit") != MAX_SELECTION_NODES
        or optimizer.get("time_check_interval_nodes")
        != SELECTION_TIME_CHECK_INTERVAL_NODES
        or optimizer.get("runtime_limit_seconds") != MAX_SELECTION_SECONDS
        or optimizer.get("working_memory_limit_bytes") != MAX_SELECTION_WORKING_BYTES
        or optimizer.get("nodes") != verifier_nodes
        or optimizer.get("nodes", MAX_SELECTION_NODES + 1) > MAX_SELECTION_NODES
        or float(optimizer.get("elapsed_seconds", float("inf"))) > MAX_SELECTION_SECONDS
        or int(optimizer.get("peak_working_bytes", MAX_SELECTION_WORKING_BYTES + 1))
        > MAX_SELECTION_WORKING_BYTES
        or verifier_elapsed > MAX_SELECTION_SECONDS
        or verifier_peak > MAX_SELECTION_WORKING_BYTES
        or verifier_input_bytes > MAX_SELECTION_INPUT_BYTES
    ):
        raise ValueError("optimizer_resource_or_independent_run_mismatch")
    return {
        "status": "verified",
        "openings": len(openings),
        "games": len(games),
        "candidate_count": len(candidate_rows),
        "selected_count": len(chosen),
        "stone_bin_counts": dict(expected_counts),
        "opening_clusters": len(by_opening),
        "game_clusters": len(by_game),
        "training_disjoint": False,
        "sampling_contract": selected["sampling_contract"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    print(
        json.dumps(verify_bundle(parser.parse_args().bundle), sort_keys=True, indent=2)
    )
