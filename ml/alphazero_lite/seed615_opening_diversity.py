"""Read-only Seed 615 source-opening-diversity feasibility preparation.

This module authenticates the Seed 599 source corpus and frozen Seed 604 B512
membership, then selects a phase-matched, opening-group-balanced candidate.
It has no training or arena execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed599_replay_coverage as seed599
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/data/seed615-source-opening-diversity"
SEED = 615
PHASE_COUNTS = {"above_32": 512, "17_to_32": 1024, "at_most_16": 512}
B512_MEMBERSHIP_SHA256 = (
    "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516"
)
B512_CHECKPOINT_SHA256 = (
    "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"
)
HELDOUT_SHA256 = "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8"
SEED604_ASSIGNMENT_SHA256 = (
    "bcbccf36657092f4320d86afaf7da7b475db059f3172259157c4f9d0298e65c7"
)
EVALUATION_SEED = 6150615
EVALUATION_OPENING_COUNT = 1024


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        b"\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            for row in rows
        )
        + b"\n"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_opening(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source"]), str(row["opening_id"])


def group_rank(phase: str, group: tuple[str, str]) -> str:
    return hashlib.sha256(
        f"seed615-group:{SEED}:{phase}:{group[0]}:{group[1]}".encode()
    ).hexdigest()


def state_rank(phase: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"seed615-state:{SEED}:{phase}:{row['canonical_identity']}".encode()
    ).hexdigest()


def select_balanced(
    eligible: dict[str, dict[str, Any]], quotas: dict[str, int] = PHASE_COUNTS
) -> list[dict[str, Any]]:
    """Round-robin phase-specific rows over deterministically ordered groups.

    Each group contributes at most one row per round. Within a group, rows are
    ordered by a phase- and canonical-identity-derived SHA-256 rank. Groups are
    ordered by a phase- and group-identity-derived SHA-256 rank. Exhausted
    groups are skipped on later rounds. A phase fails closed unless the eligible
    population can fill its exact quota.
    """
    phase_groups: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        phase: defaultdict(list) for phase in quotas
    }
    for row in eligible.values():
        phase = str(row["phase"])
        if phase not in phase_groups:
            raise ValueError(f"unknown_phase:{phase}")
        phase_groups[phase][source_opening(row)].append(row)

    selected: list[dict[str, Any]] = []
    for phase, quota in quotas.items():
        groups = phase_groups[phase]
        available = sum(map(len, groups.values()))
        if available < quota:
            raise ValueError(
                f"insufficient_phase_positions:{phase}:{available}:{quota}"
            )
        ordered_groups = sorted(
            groups, key=lambda group: (group_rank(phase, group), group)
        )
        ordered_rows = {
            group: sorted(
                rows,
                key=lambda row: (state_rank(phase, row), row["canonical_identity"]),
            )
            for group, rows in groups.items()
        }
        picked = 0
        round_index = 0
        while picked < quota:
            advanced = False
            for group in ordered_groups:
                rows = ordered_rows[group]
                if round_index < len(rows) and picked < quota:
                    selected.append(rows[round_index])
                    picked += 1
                    advanced = True
            if not advanced:
                raise ValueError(f"group_capacity_exhausted:{phase}:{picked}:{quota}")
            round_index += 1
    selected.sort(key=lambda row: (row["phase"], row["canonical_identity"]))
    if Counter(row["phase"] for row in selected) != Counter(quotas):
        raise ValueError("phase_quota_mismatch")
    if len({row["canonical_identity"] for row in selected}) != len(selected):
        raise ValueError("duplicate_canonical_identity")
    if len({row["float32_identity"] for row in selected}) != len(selected):
        raise ValueError("duplicate_float32_identity")
    return selected


def authenticated_population(root: Path = ROOT) -> dict[str, Any]:
    """Re-authenticate #580/#585 and reproduce the original provenance split."""
    seed599.read_only_preflight(root)
    records, source_verification = seed599.authenticated_records(root)
    groups = sorted({source_opening(row) for row in records})
    heldout_groups = set(
        sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"seed599-heldout:{seed599.SEED}:{group[0]}:{group[1]}".encode()
            ).hexdigest(),
        )[: (len(groups) + 4) // 5]
    )
    eligible, heldout = seed599.split_provenance_records(records, heldout_groups)
    heldout_path = root / "docs/data/seed599-replay-coverage/heldout-membership.json"
    heldout_manifest = json.loads(heldout_path.read_bytes())
    if sha256_bytes(canonical_bytes(heldout_manifest)) != HELDOUT_SHA256:
        raise ValueError("heldout_manifest_identity_mismatch")
    if canonical_bytes(heldout) != canonical_bytes(heldout_manifest):
        raise ValueError("heldout_membership_reconstruction_mismatch")

    b512_path = root / "docs/data/seed599-replay-coverage/treatment-membership.json"
    b512 = json.loads(b512_path.read_bytes())
    b512_identity = sha256_bytes(canonical_bytes(b512))
    if b512_identity != B512_MEMBERSHIP_SHA256:
        raise ValueError("b512_membership_identity_mismatch")
    registration = json.loads(
        (root / "docs/data/seed604-replay-optimization/registration.json").read_bytes()
    )
    if registration["treatment"]["membership_sha256"] != B512_MEMBERSHIP_SHA256:
        raise ValueError("seed604_b512_registration_mismatch")
    checkpoint = root / registration["control"]["immutable_seed599_checkpoint"]
    if sha256_bytes(checkpoint.read_bytes()) != B512_CHECKPOINT_SHA256:
        raise ValueError("b512_checkpoint_identity_mismatch")
    seed599_openings = json.loads(
        (
            root / "docs/data/seed599-replay-coverage/evaluation-openings.json"
        ).read_bytes()
    )
    prior_canonical = {str(row["canonical_identity"]) for row in seed599_openings}
    prior_float32 = {str(row["float32_identity"]) for row in seed599_openings}
    seed604_manifest_bytes = (
        root
        / "docs/data/seed604-replay-optimization/evaluation-preparation/assignment-manifest.json"
    ).read_bytes()
    if sha256_bytes(seed604_manifest_bytes) != SEED604_ASSIGNMENT_SHA256:
        raise ValueError("seed604_assignment_manifest_identity_mismatch")
    seed604_manifest = json.loads(seed604_manifest_bytes)
    seen_openings: set[str] = set()
    for assignment in seed604_manifest["assignments"]:
        opening_id = str(assignment["opening_id"])
        if opening_id in seen_openings:
            continue
        seen_openings.add(opening_id)
        game = initial_game()
        for move in assignment["prefix_moves"]:
            legal = game.possible_moves()
            if int(move) not in legal:
                raise ValueError("seed604_registered_opening_prefix_illegal")
            game.move(game.pit_index(int(move)))
        state = game.to_state()
        canonical = seed599.canonical_identity(state)
        if canonical != assignment["opening_identity"]:
            raise ValueError("seed604_registered_opening_identity_mismatch")
        prior_canonical.add(canonical)
        prior_float32.add(seed599.production_float32_identity(state))
    return {
        "records": records,
        "eligible": eligible,
        "heldout": heldout,
        "b512": b512,
        "heldout_groups": heldout_groups,
        "source_verification": source_verification,
        "b512_membership_sha256": b512_identity,
        "b512_checkpoint_sha256": B512_CHECKPOINT_SHA256,
        "prior_evaluation_opening_canonical": prior_canonical,
        "prior_evaluation_opening_float32": prior_float32,
        "seed604_assignment_manifest_sha256": SEED604_ASSIGNMENT_SHA256,
        "eligible_inventory_sha256": sha256_bytes(
            canonical_bytes(
                sorted(
                    (row["canonical_identity"], row["source"], row["opening_id"])
                    for row in eligible.values()
                )
            )
        ),
    }


def entropy(target: list[float]) -> float:
    return -sum(value * math.log(value) for value in target if value > 0)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(source_opening(row) for row in rows)
    count = len(rows)
    shares = sorted((value / count for value in groups.values()), reverse=True)
    target_entropy = [entropy(row["target"]) for row in rows]
    support = Counter(sum(value > 0 for value in row["target"]) for row in rows)
    teacher = Counter(
        row["teacher_evidence"].get("teacher_source", "unknown") for row in rows
    )
    phase_stats: dict[str, Any] = {}
    for phase in PHASE_COUNTS:
        subset = [row for row in rows if row["phase"] == phase]
        phase_stats[phase] = {
            "n": len(subset),
            "active_pit_stones": {
                str(key): value
                for key, value in sorted(
                    Counter(int(row["stone_count"]) for row in subset).items()
                )
            },
            "legal_move_count": {
                str(key): value
                for key, value in sorted(
                    Counter(len(row["legal_moves"]) for row in subset).items()
                )
            },
            "empty_pits": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        sum(
                            value == 0
                            for value in list(row["canonical_state"]["player_pits"])
                            + list(row["canonical_state"]["opponent_pits"])
                        )
                        for row in subset
                    ).items()
                )
            },
            "teacher_source": dict(
                sorted(
                    Counter(
                        row["teacher_evidence"].get("teacher_source", "unknown")
                        for row in subset
                    ).items()
                )
            ),
            "target_support_size": dict(
                sorted(
                    Counter(
                        sum(value > 0 for value in row["target"]) for row in subset
                    ).items()
                )
            ),
            "mean_target_entropy_nats": float(
                np.mean([entropy(row["target"]) for row in subset])
            ),
            "mean_max_target_probability": float(
                np.mean([max(row["target"]) for row in subset])
            ),
        }
    return {
        "states": count,
        "phase_counts": dict(Counter(row["phase"] for row in rows)),
        "opening_groups": {
            "distinct": len(groups),
            "effective_number_inverse_simpson": float(
                1.0 / sum((value / count) ** 2 for value in groups.values())
            ),
            "largest_group_share": shares[0],
            "top_10_group_share": sum(shares[:10]),
            "per_group_counts": [
                {"source": source, "opening_id": opening, "n": value}
                for (source, opening), value in sorted(groups.items())
            ],
        },
        "active_pit_stones_by_phase": {
            phase: phase_stats[phase]["active_pit_stones"] for phase in PHASE_COUNTS
        },
        "legal_move_count_by_phase": {
            phase: phase_stats[phase]["legal_move_count"] for phase in PHASE_COUNTS
        },
        "empty_pit_count_by_phase": {
            phase: phase_stats[phase]["empty_pits"] for phase in PHASE_COUNTS
        },
        "teacher_source_counts": dict(sorted(teacher.items())),
        "target_support_size_counts": dict(sorted(support.items())),
        "mean_target_entropy_nats": float(np.mean(target_entropy)),
        "mean_max_target_probability": float(
            np.mean([max(row["target"]) for row in rows])
        ),
        "by_phase": phase_stats,
    }


def eligible_by_group(eligible: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (row["source"], row["opening_id"], row["phase"]) for row in eligible.values()
    )
    return [
        {"source": source, "opening_id": opening, "phase": phase, "eligible_states": n}
        for (source, opening, phase), n in sorted(counts.items())
    ]


def initial_game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def generate_evaluation_openings(
    authenticated: dict[str, Any], candidate: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create unique legal random 12-ply roots outside registered prior cohorts."""
    training = (
        list(authenticated["eligible"].values())
        + list(authenticated["heldout"])
        + candidate
    )
    blocked_canonical = {row["canonical_identity"] for row in training} | authenticated[
        "prior_evaluation_opening_canonical"
    ]
    blocked_float32 = {row["float32_identity"] for row in training} | authenticated[
        "prior_evaluation_opening_float32"
    ]
    rng = random.Random(EVALUATION_SEED)
    canonical_seen: set[str] = set()
    float32_seen: set[str] = set()
    openings: list[dict[str, Any]] = []
    attempt = 0
    while len(openings) < EVALUATION_OPENING_COUNT:
        attempt += 1
        game = initial_game()
        prefix: list[int] = []
        for _ in range(12):
            legal = game.possible_moves()
            if not legal or game.over():
                break
            move = int(rng.choice(legal))
            prefix.append(move)
            game.move(game.pit_index(move))
        if len(prefix) != 12 or game.over():
            continue
        state = game.to_state()
        canonical = seed599.canonical_identity(state)
        float32 = seed599.production_float32_identity(state)
        if (
            canonical in blocked_canonical
            or float32 in blocked_float32
            or canonical in canonical_seen
            or float32 in float32_seen
        ):
            continue
        canonical_seen.add(canonical)
        float32_seen.add(float32)
        openings.append(
            {
                "opening_id": f"seed615-opening-{len(openings):04d}",
                "prefix_moves": prefix,
                "state": state,
                "canonical_identity": canonical,
                "float32_identity": float32,
                "generation_attempt": attempt,
            }
        )
    return openings


def game_seed(opening_id: str, candidate: str, seat: int) -> int:
    material = f"seed615\0{opening_id}\0{candidate}\0{seat}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def generate_evaluation_assignments(
    openings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze each candidate/seat game assignment without executing games."""
    assignments = []
    for index, opening in enumerate(openings):
        for candidate in ("B512", "S615"):
            for seat in (0, 1):
                assignments.append(
                    {
                        "game_id": f"seed615-{candidate}-{index:04d}-p{seat}",
                        "game_seed": game_seed(opening["opening_id"], candidate, seat),
                        "opening_id": opening["opening_id"],
                        "opening_index": index,
                        "opening_identity": opening["canonical_identity"],
                        "candidate": candidate,
                        "candidate_seat": seat,
                        "opponent": "P-E4",
                    }
                )
    return assignments


def build(root: Path = ROOT) -> dict[str, Any]:
    authenticated = authenticated_population(root)
    eligible = authenticated["eligible"]
    candidate = select_balanced(eligible)
    evaluation_openings = generate_evaluation_openings(authenticated, candidate)
    evaluation_assignments = generate_evaluation_assignments(evaluation_openings)
    b512 = authenticated["b512"]
    heldout = authenticated["heldout"]
    seed599.validate_training_heldout_disjointness(candidate, b512, heldout)
    for row in candidate:
        seed599.validate_target_record(row)
    candidate_sha = sha256_bytes(canonical_bytes(candidate))
    baseline = summarize(b512)
    treatment = summarize(candidate)
    b512_canonical = {row["canonical_identity"] for row in b512}
    candidate_canonical = {row["canonical_identity"] for row in candidate}
    b512_float32 = {row["float32_identity"] for row in b512}
    candidate_float32 = {row["float32_identity"] for row in candidate}
    diversity_increased = (
        treatment["opening_groups"]["effective_number_inverse_simpson"]
        > baseline["opening_groups"]["effective_number_inverse_simpson"]
    )
    effective_gain = (
        treatment["opening_groups"]["effective_number_inverse_simpson"]
        - baseline["opening_groups"]["effective_number_inverse_simpson"]
    )
    return {
        "schema": "seed615-source-opening-diversity-feasibility-v1",
        "status": "read-only-preparation-no-training-or-evaluation",
        "base_commit": "52895079610da6d535809a454d2b2502548068f0",
        "authentication": {
            "seed599_read_only_preflight": "passed",
            "source_verification": authenticated["source_verification"],
            "eligible_inventory_sha256": authenticated["eligible_inventory_sha256"],
            "eligible_unique_states": len(eligible),
            "eligible_by_group_and_phase": eligible_by_group(eligible),
            "heldout_states": len(heldout),
            "heldout_membership_sha256": HELDOUT_SHA256,
            "b512_membership_sha256": authenticated["b512_membership_sha256"],
            "b512_checkpoint_sha256": authenticated["b512_checkpoint_sha256"],
            "b512_membership_rows": len(b512),
        },
        "candidate": {
            "algorithm": (
                "For each original phase quota independently, rank source opening groups by "
                "SHA256('seed615-group:615:<phase>:<source>:<opening_id>'), rank each group's "
                "eligible states by SHA256('seed615-state:615:<phase>:<canonical_identity>'), "
                "then take one state per group per round until the phase quota is met. Skip "
                "groups exhausted in prior rounds; fail closed if total phase availability is "
                "below quota. Final manifest order is (phase, canonical_identity)."
            ),
            "selection_seed": SEED,
            "membership_sha256": candidate_sha,
            "membership_rows": len(candidate),
            "canonical_unique": len({row["canonical_identity"] for row in candidate}),
            "production_float32_unique": len(
                {row["float32_identity"] for row in candidate}
            ),
            "canonical_and_float32_disjoint_from_heldout": True,
            "phase_quotas": dict(PHASE_COUNTS),
            "original_teacher_targets_retained": True,
            "membership": candidate,
        },
        "comparison": {
            "b512": baseline,
            "candidate": treatment,
            "membership_overlap": {
                "canonical_shared": len(b512_canonical & candidate_canonical),
                "canonical_b512_only": len(b512_canonical - candidate_canonical),
                "canonical_candidate_only": len(candidate_canonical - b512_canonical),
                "production_float32_shared": len(b512_float32 & candidate_float32),
                "production_float32_b512_only": len(b512_float32 - candidate_float32),
                "production_float32_candidate_only": len(
                    candidate_float32 - b512_float32
                ),
            },
            "effective_opening_group_number_gain": effective_gain,
            "effective_opening_group_number_ratio": (
                treatment["opening_groups"]["effective_number_inverse_simpson"]
                / baseline["opening_groups"]["effective_number_inverse_simpson"]
            ),
            "effective_opening_group_diversity_increased": diversity_increased,
            "recommendation": "train" if diversity_increased else "do_not_train",
        },
        "prospective_evaluation_cohort": {
            "status": "registered-preparation-only-not-authorized",
            "seed": EVALUATION_SEED,
            "opening_count": len(evaluation_openings),
            "games": len(evaluation_openings) * 4,
            "generation": "Exactly 12 legal moves sampled uniformly from legal actions using Python random.Random(6150615); reject terminal resulting states and canonical or production-float32 overlap with all eligible/held-out source states and authenticated Seed 599/604 evaluation opening roots; require uniqueness in both identities",
            "opening_manifest_sha256": sha256_bytes(
                canonical_bytes(evaluation_openings)
            ),
            "openings": evaluation_openings,
            "assignment_count": len(evaluation_assignments),
            "assignment_manifest_sha256": sha256_bytes(
                canonical_bytes(evaluation_assignments)
            ),
            "assignments": evaluation_assignments,
        },
    }


def write(root: Path = ROOT) -> dict[str, Any]:
    result = build(root)
    destination = root / "docs/data/seed615-source-opening-diversity"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidate-membership.jsonl").write_bytes(
        jsonl_bytes(result["candidate"]["membership"])
    )
    (destination / "evaluation-openings.jsonl").write_bytes(
        jsonl_bytes(result["prospective_evaluation_cohort"]["openings"])
    )
    (destination / "evaluation-assignments.jsonl").write_bytes(
        jsonl_bytes(result["prospective_evaluation_cohort"]["assignments"])
    )
    compact = json.loads(json.dumps(result))
    compact["candidate"].pop("membership")
    compact["prospective_evaluation_cohort"].pop("openings")
    compact["prospective_evaluation_cohort"].pop("assignments")
    (destination / "feasibility.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n"
    )
    return compact


def verify(root: Path = ROOT) -> dict[str, Any]:
    expected = build(root)
    destination = root / "docs/data/seed615-source-opening-diversity"
    membership = read_jsonl(destination / "candidate-membership.jsonl")
    openings = read_jsonl(destination / "evaluation-openings.jsonl")
    assignments = read_jsonl(destination / "evaluation-assignments.jsonl")
    report = json.loads((destination / "feasibility.json").read_bytes())
    if canonical_bytes(membership) != canonical_bytes(
        expected["candidate"]["membership"]
    ):
        raise ValueError("candidate_membership_mismatch")
    if canonical_bytes(openings) != canonical_bytes(
        expected["prospective_evaluation_cohort"]["openings"]
    ):
        raise ValueError("evaluation_opening_membership_mismatch")
    if canonical_bytes(assignments) != canonical_bytes(
        expected["prospective_evaluation_cohort"]["assignments"]
    ):
        raise ValueError("evaluation_assignment_membership_mismatch")
    expected_report = json.loads(json.dumps(expected))
    expected_report["candidate"].pop("membership")
    expected_report["prospective_evaluation_cohort"].pop("openings")
    expected_report["prospective_evaluation_cohort"].pop("assignments")
    if canonical_bytes(report) != canonical_bytes(expected_report):
        raise ValueError("feasibility_report_mismatch")
    return {
        "passed": True,
        "training_or_evaluation_started": False,
        "candidate_states": len(membership),
        "candidate_membership_sha256": expected["candidate"]["membership_sha256"],
        "evaluation_openings": len(openings),
        "evaluation_assignments": len(assignments),
        "evaluation_opening_manifest_sha256": expected["prospective_evaluation_cohort"][
            "opening_manifest_sha256"
        ],
        "recommendation": expected["comparison"]["recommendation"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    result = write() if args.command == "prepare" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
