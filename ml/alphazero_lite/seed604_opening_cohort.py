"""Frozen deterministic Seed 604 evaluation-opening cohort generator."""

from __future__ import annotations

import hashlib
import argparse
import json
import random
import subprocess
from collections import Counter
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

SEED = 604
COUNT = 1024
PLIES = 12
INITIAL_STATE = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
PUBLICATION = Path("docs/data/seed604-evaluation-opening-cohort-v4")
CONTRACT = {
    "schema": "seed604-opening-generation-contract-v1",
    "seed": SEED,
    "target_count": COUNT,
    "opening_plies": PLIES,
    "reference": "Seed 599 generate_openings semantics in seed599_replay_coverage.py",
    "rng": "CPython random.Random(604); each candidate uses random.Random(master.randrange(2**63))",
    "sampling": "local_rng.choice(game.possible_moves()) exactly once per legal ply",
    "initial_state": INITIAL_STATE,
    "rules_source": "ml/alphazero_lite/kalah_rules.py KalahGame; relative moves mapped with pit_index and applied with move",
    "terminal_rule": "Reject a candidate if terminal before completion, at any ply before ply 12, or after ply 12; require exactly 12 legal moves",
    "identities": "canonical_identity and production_float32_identity from seed551_controlled_policy_strength; validate through authenticated published bounded policy interface",
    "ordering": "Candidate indices start at 1; process serially; retain first 1024 candidates passing ordered rejection checks",
    "rejection_order": [
        "terminal_or_incomplete",
        "canonical_exclusion",
        "float32_exclusion",
        "canonical_duplicate",
        "float32_duplicate",
    ],
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sample_candidate(
    local: random.Random,
) -> tuple[list[int], dict[str, Any], str | None]:
    game = KalahGame.from_state(dict(INITIAL_STATE))
    prefix: list[int] = []
    if game.over():
        return prefix, game.to_state(), "terminal_or_incomplete"
    for _ in range(PLIES):
        if game.over():
            return prefix, game.to_state(), "terminal_or_incomplete"
        legal = game.possible_moves()
        if not legal:
            return prefix, game.to_state(), "terminal_or_incomplete"
        move = int(local.choice(legal))
        prefix.append(move)
        if not game.move(game.pit_index(move)):
            raise ValueError("legal_move_rejected")
        if game.over() and len(prefix) < PLIES:
            return prefix, game.to_state(), "terminal_or_incomplete"
    if len(prefix) != PLIES or game.over():
        return prefix, game.to_state(), "terminal_or_incomplete"
    return prefix, game.to_state(), None


def generate(root: Path = ROOT) -> dict[str, Any]:
    """Generate a manifest in memory; never creates or overwrites files."""
    policy = load_authenticated_published_policy(root)
    master = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    float32_seen: set[str] = set()
    rejected: Counter[str] = Counter()
    candidate = 0
    while len(selected) < COUNT:
        candidate += 1
        prefix, state, replay_rejection = _sample_candidate(
            random.Random(master.randrange(2**63))
        )
        if replay_rejection:
            rejected[replay_rejection] += 1
            continue
        canonical = canonical_identity(state)
        float32 = production_float32_identity(state)
        try:
            authenticated = validate_candidate_against_published_policy(state, policy)
        except ValueError as error:
            message = str(error)
            if message == "candidate_canonical_exclusion_collision":
                rejected["canonical_exclusion"] += 1
                continue
            if message == "candidate_float32_exclusion_collision":
                rejected["float32_exclusion"] += 1
                continue
            raise
        if authenticated != {"canonical": canonical, "float32": float32}:
            raise ValueError("authenticated_identity_mismatch")
        if canonical in canonical_seen:
            rejected["canonical_duplicate"] += 1
            continue
        if float32 in float32_seen:
            rejected["float32_duplicate"] += 1
            continue
        selected.append(
            {
                "opening_id": f"seed604-opening-{len(selected):04d}",
                "candidate_index": candidate,
                "prefix_moves": prefix,
                "state": state,
                "canonical_identity": canonical,
                "float32_identity": float32,
            }
        )
        canonical_seen.add(canonical)
        float32_seen.add(float32)
    return {
        "schema": "seed604-evaluation-opening-manifest-v1",
        "generation_contract": CONTRACT,
        "openings": selected,
        "accounting": {
            "candidate_count": candidate,
            "accepted_count": COUNT,
            "rejected_by_reason": {
                reason: rejected[reason] for reason in CONTRACT["rejection_order"]
            },
        },
    }


def publication_payloads(root: Path = ROOT) -> dict[str, bytes]:
    """Build create-only files; caller must have frozen/committed this module."""
    manifest = generate(root)
    source = Path(__file__).resolve()
    source_hash = sha256(source.read_bytes())
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    policy_path = (
        root / "docs/data/seed604-replay-optimization/bounded-exclusion-policy.json"
    )
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw)
    if policy["observed_overlap_accounting"]["bounded_union_per_scheme"] != 537698:
        raise ValueError("bounded_union_count_mismatch")
    union_digests = policy["observed_overlap_accounting"][
        "bounded_union_identity_set_sha256"
    ]
    contract_raw = json.dumps(CONTRACT, sort_keys=True, indent=2).encode() + b"\n"
    manifest_raw = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    ids = {
        "canonical": {row["canonical_identity"] for row in manifest["openings"]},
        "production_float32": {row["float32_identity"] for row in manifest["openings"]},
    }
    registration = {
        "schema": "seed604-opening-cohort-registration-v1",
        "source_commit": source_commit,
        "generator_source_sha256": source_hash,
        "rules_source_sha256": sha256(
            (root / "ml/alphazero_lite/kalah_rules.py").read_bytes()
        ),
        "identity_source_sha256": sha256(
            (
                root / "ml/alphazero_lite/seed551_controlled_policy_strength.py"
            ).read_bytes()
        ),
        "verifier_source_sha256": sha256(
            (root / "ml/alphazero_lite/verify_seed604_opening_cohort.py").read_bytes()
        ),
        "contract_sha256": sha256(contract_raw),
        "manifest_sha256": sha256(manifest_raw),
        "exclusion_policy_sha256": sha256(policy_raw),
        "exclusion_union_distinct": 537698,
        "exclusion_union_sha256": union_digests,
        "identity_sets": {
            name: sha256("\n".join(sorted(values)).encode())
            for name, values in ids.items()
        },
    }
    verifier_path = root / "ml/alphazero_lite/verify_seed604_opening_cohort.py"
    files: dict[str, Any] = {
        "generation-contract.json": CONTRACT,
        "opening-manifest.json": manifest,
        "registration.json": registration,
    }
    # Policy source is pinned by the registration and re-authenticated at verify time.
    payloads = {
        name: json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
        for name, value in files.items()
    }
    payloads["authenticated-exclusion-policy.json"] = policy_raw
    payloads["generator-source.py"] = source.read_bytes()
    payloads["verifier-source.py"] = verifier_path.read_bytes()
    return payloads


def publish(root: Path = ROOT) -> None:
    payloads = publication_payloads(root)
    destination = root / PUBLICATION
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (destination / name).open("xb") as stream:
            stream.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish",))
    args = parser.parse_args()
    if args.action == "publish":
        publish()
