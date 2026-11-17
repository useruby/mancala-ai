"""Independent, read-only verifier for the Seed 604 opening cohort."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.seed604_bounded_exclusions import (
    ROOT,
    load_authenticated_published_policy,
    validate_candidate_against_published_policy,
)

OUTPUT = Path("docs/data/seed604-evaluation-opening-cohort-v5")
SOURCE = Path("ml/alphazero_lite/seed604_opening_cohort.py")
VERIFIER = Path(__file__).relative_to(ROOT)
INITIAL = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_digest(raw: bytes, expected: str, label: str) -> None:
    if digest(raw) != expected:
        raise ValueError(f"{label}_hash_mismatch")


def require_distinct_identities(rows: list[dict[str, Any]]) -> None:
    canonical_values = [row["canonical_identity"] for row in rows]
    float32_values = [row["float32_identity"] for row in rows]
    if len(set(canonical_values)) != len(rows):
        raise ValueError("duplicate_canonical_root")
    if len(set(float32_values)) != len(rows):
        raise ValueError("duplicate_float32_root")


def _bytes(root: Path, relative: str) -> bytes:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing_evidence:{relative}")
    return path.read_bytes()


def _independent_replay(prefix: list[int]) -> dict[str, Any]:
    """Replay without production candidate-generation helpers."""
    if len(prefix) != 12:
        raise ValueError("prefix_length_mismatch")
    position = KalahGame.from_state(dict(INITIAL))
    for ply, relative_move in enumerate(prefix, start=1):
        legal_moves = position.possible_moves()
        if position.over() or relative_move not in legal_moves:
            raise ValueError("illegal_prefix_move")
        absolute_move = relative_move + 6 * position.current_player
        if not position.move(absolute_move):
            raise ValueError("replay_rejected_legal_move")
        if position.over() and ply < 12:
            raise ValueError("terminal_before_prefix_end")
    if position.over():
        raise ValueError("terminal_prefix_end")
    return position.to_state()


def _independent_generate(root: Path, policy: Any) -> dict[str, Any]:
    """Reproduce Seed 599 sampling semantics in a separate verifier loop."""
    master_rng = random.Random(604)
    accepted: list[dict[str, Any]] = []
    used_canonical: set[str] = set()
    used_float32: set[str] = set()
    rejected = {
        "terminal_or_incomplete": 0,
        "canonical_exclusion": 0,
        "float32_exclusion": 0,
        "canonical_duplicate": 0,
        "float32_duplicate": 0,
    }
    candidate_index = 0
    while len(accepted) != 1024:
        candidate_index += 1
        local_rng = random.Random(master_rng.randrange(2**63))
        game = KalahGame.from_state(dict(INITIAL))
        prefix: list[int] = []
        failed = False
        while len(prefix) < 12:
            available = game.possible_moves()
            if game.over() or not available:
                failed = True
                break
            choice = int(local_rng.choice(available))
            prefix.append(choice)
            if not game.move(game.pit_index(choice)):
                raise ValueError("verifier_sample_move_rejected")
            if game.over() and len(prefix) != 12:
                failed = True
                break
        final_state = game.to_state()
        if failed or len(prefix) != 12 or game.over():
            rejected["terminal_or_incomplete"] += 1
            continue
        canonical = canonical_identity(final_state)
        float32 = production_float32_identity(final_state)
        try:
            authenticated = validate_candidate_against_published_policy(
                final_state, policy
            )
        except ValueError as error:
            reason = str(error)
            if reason == "candidate_canonical_exclusion_collision":
                rejected["canonical_exclusion"] += 1
                continue
            if reason == "candidate_float32_exclusion_collision":
                rejected["float32_exclusion"] += 1
                continue
            raise
        if authenticated != {"canonical": canonical, "float32": float32}:
            raise ValueError("verifier_authenticated_identity_mismatch")
        if canonical in used_canonical:
            rejected["canonical_duplicate"] += 1
            continue
        if float32 in used_float32:
            rejected["float32_duplicate"] += 1
            continue
        accepted.append(
            {
                "opening_id": f"seed604-opening-{len(accepted):04d}",
                "candidate_index": candidate_index,
                "prefix_moves": prefix,
                "state": final_state,
                "canonical_identity": canonical,
                "float32_identity": float32,
            }
        )
        used_canonical.add(canonical)
        used_float32.add(float32)
    return {
        "schema": "seed604-evaluation-opening-manifest-v1",
        "generation_contract": json.loads(
            _bytes(root, str(OUTPUT / "generation-contract.json"))
        ),
        "openings": accepted,
        "accounting": {
            "candidate_count": candidate_index,
            "accepted_count": 1024,
            "rejected_by_reason": rejected,
        },
    }


def verify(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    policy = load_authenticated_published_policy(root)
    base = str(OUTPUT)
    registration = json.loads(_bytes(root, f"{base}/registration.json"))
    manifest_raw = _bytes(root, f"{base}/opening-manifest.json")
    manifest = json.loads(manifest_raw)
    contract_raw = _bytes(root, f"{base}/generation-contract.json")
    policy_raw = _bytes(root, f"{base}/authenticated-exclusion-policy.json")
    actual_policy = _bytes(
        root, "docs/data/seed604-replay-optimization/bounded-exclusion-policy.json"
    )
    if policy_raw != actual_policy:
        raise ValueError("published_policy_copy_mismatch")
    require_digest(
        policy_raw, registration.get("exclusion_policy_sha256", ""), "exclusion_policy"
    )
    require_digest(manifest_raw, registration.get("manifest_sha256", ""), "manifest")
    require_digest(contract_raw, registration.get("contract_sha256", ""), "contract")
    source_raw = _bytes(root, str(SOURCE))
    if digest(source_raw) != registration.get("generator_source_sha256"):
        raise ValueError("generator_source_hash_mismatch")
    if source_raw != _bytes(root, f"{base}/generator-source.py"):
        raise ValueError("generator_snapshot_mismatch")
    verifier_raw = _bytes(root, str(VERIFIER))
    if verifier_raw != _bytes(root, f"{base}/verifier-source.py"):
        raise ValueError("verifier_snapshot_mismatch")
    if digest(verifier_raw) != registration.get("verifier_source_sha256"):
        raise ValueError("verifier_source_hash_mismatch")
    source_commit = registration.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("source_commit_missing")
    # Ensure the frozen source commit is present and is an ancestor of this checkout.
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rules_sha = digest(_bytes(root, "ml/alphazero_lite/kalah_rules.py"))
    identity_sha = digest(
        _bytes(root, "ml/alphazero_lite/seed551_controlled_policy_strength.py")
    )
    if rules_sha != registration.get("rules_source_sha256"):
        raise ValueError("rules_source_hash_mismatch")
    if identity_sha != registration.get("identity_source_sha256"):
        raise ValueError("identity_source_hash_mismatch")
    bounded = policy.document["observed_overlap_accounting"]
    if bounded["bounded_union_per_scheme"] != 537698:
        raise ValueError("bounded_union_count_mismatch")
    if bounded["bounded_union_identity_set_sha256"] != registration.get(
        "exclusion_union_sha256"
    ):
        raise ValueError("exclusion_union_digest_mismatch")
    if digest(contract_raw) != registration["contract_sha256"]:
        raise ValueError("generation_contract_hash_mismatch")
    if manifest["generation_contract"] != json.loads(contract_raw):
        raise ValueError("manifest_contract_mismatch")
    rows = manifest.get("openings")
    if not isinstance(rows, list) or len(rows) != 1024:
        raise ValueError("opening_count_mismatch")
    require_distinct_identities(rows)
    canonical_set: set[str] = set()
    float32_set: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("opening_id") != f"seed604-opening-{index:04d}":
            raise ValueError("opening_order_or_duplicate_mismatch")
        replayed = _independent_replay(row.get("prefix_moves", []))
        if replayed != row.get("state"):
            raise ValueError("replayed_state_mismatch")
        canonical = canonical_identity(replayed)
        float32 = production_float32_identity(replayed)
        if canonical != row.get("canonical_identity") or float32 != row.get(
            "float32_identity"
        ):
            raise ValueError("opening_identity_mismatch")
        validate_candidate_against_published_policy(replayed, policy)
        if canonical in canonical_set or float32 in float32_set:
            raise ValueError("duplicate_opening_root")
        canonical_set.add(canonical)
        float32_set.add(float32)
    reproduced = _independent_generate(root, policy)
    if reproduced != manifest:
        raise ValueError("deterministic_generation_or_accounting_mismatch")
    id_sets = registration.get("identity_sets", {})
    for name, key in (
        ("canonical", "canonical_identity"),
        ("production_float32", "float32_identity"),
    ):
        expected = digest("\n".join(sorted({row[key] for row in rows})).encode())
        if id_sets.get(name) != expected:
            raise ValueError(f"identity_set_hash_mismatch:{name}")
    statuses = policy.document.get("execution_status", {})
    if any(statuses.values()):
        raise ValueError("unexpected_execution_artifact_status")
    return {
        "passed": True,
        "opening_count": len(rows),
        "accounting": manifest["accounting"],
        "manifest_sha256": registration["manifest_sha256"],
        "exclusion_union_sha256": registration["exclusion_union_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
