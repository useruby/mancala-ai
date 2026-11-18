"""Independent read-only verifier for the prepared Seed 615 memberships."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed599_replay_coverage as seed599
from ml.alphazero_lite import seed615_opening_diversity as seed615
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BOOTSTRAP_SPEC = {
    "method": "paired opening-cluster percentile bootstrap",
    "resamples": 20000,
    "seed": 615,
    "random_generator": "numpy.random.default_rng(615)",
    "quantile_method": "numpy.quantile(method='linear')",
    "resampling": "For each replicate sample 1,024 opening-cluster indices with replacement; apply those opening indices jointly to paired candidate/comparison score vectors and retain both seat-swapped outcomes within each selected opening.",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rank(prefix: str, phase: str, key: str) -> str:
    return hashlib.sha256(f"{prefix}:615:{phase}:{key}".encode()).hexdigest()


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


def validate_opening_prefix(
    prefix_moves: Any,
    *,
    game_factory: Callable[[], KalahGame] | None = None,
) -> dict[str, Any]:
    """Replay one registered opening and require exactly 12 legal nonterminal plies."""
    if not isinstance(prefix_moves, list) or len(prefix_moves) != 12:
        raise ValueError("evaluation_opening_prefix_length_mismatch")
    if game_factory is None:
        game_factory = initial_game
    game = game_factory()
    for move in prefix_moves:
        if game.over():
            raise ValueError("evaluation_opening_terminal_before_prefix_complete")
        if type(move) is not int or move not in game.possible_moves():
            raise ValueError("evaluation_opening_prefix_illegal")
        game.move(game.pit_index(move))
    if game.over():
        raise ValueError("evaluation_opening_terminal")
    return game.to_state()


def validate_registration_contract(
    registration: dict[str, Any],
    *,
    membership_digest: str,
    opening_digest: str,
    assignment_digest: str,
) -> None:
    """Authenticate registered identities, inference choices, and no-authorization state."""
    if registration.get("seed") != 615:
        raise ValueError("registration_seed_mismatch")
    if registration.get("status") != "proposed-for-review-not-authorized":
        raise ValueError("registration_status_authorized_or_invalid")
    feasibility = registration.get("feasibility", {})
    treatment = registration.get("treatment", {})
    if (
        feasibility.get("candidate_membership")
        != "docs/data/seed615-source-opening-diversity/candidate-membership.jsonl"
        or feasibility.get("candidate_membership_sha256") != membership_digest
        or treatment.get("membership_sha256") != membership_digest
    ):
        raise ValueError("registered_candidate_membership_digest_mismatch")

    evaluation = registration.get("evaluation", {})
    cohort = evaluation.get("cohort", {})
    if evaluation.get("status") != "prospectively-registered-not-authorized":
        raise ValueError("evaluation_registration_status_invalid")
    if (
        cohort.get("path")
        != "docs/data/seed615-source-opening-diversity/evaluation-openings.jsonl"
        or cohort.get("canonical_manifest_sha256") != opening_digest
    ):
        raise ValueError("registered_evaluation_opening_digest_mismatch")
    if (
        cohort.get("assignments_path")
        != "docs/data/seed615-source-opening-diversity/evaluation-assignments.jsonl"
        or cohort.get("assignment_manifest_sha256") != assignment_digest
    ):
        raise ValueError("registered_evaluation_assignment_digest_mismatch")
    if evaluation.get("bootstrap") != EXPECTED_BOOTSTRAP_SPEC:
        raise ValueError("registered_bootstrap_spec_mismatch")

    primary = evaluation.get("primary_confirmatory_test", {})
    if (
        primary.get("interval")
        != "paired two-sided 95% opening-cluster bootstrap confidence interval"
        or primary.get("contrast")
        != "treatment minus fixed B512 seat-averaged playing score"
        or primary.get("resamples") != 20000
        or primary.get("bootstrap_seed") != 615
        or primary.get("quantiles") != [0.025, 0.975]
        or primary.get("pass_rule") != "lower endpoint strictly greater than zero"
    ):
        raise ValueError("registered_primary_bootstrap_or_threshold_mismatch")
    serving_gate = evaluation.get("separate_serving_strength_gate", {})
    if (
        serving_gate.get("bound")
        != "one-sided 95% paired opening-cluster bootstrap lower confidence bound"
        or serving_gate.get("contrast")
        != "treatment versus serving P-E4 seat-averaged score"
        or serving_gate.get("resamples") != 20000
        or serving_gate.get("bootstrap_seed") != 615
        or serving_gate.get("lower_quantile") != 0.05
        or serving_gate.get("pass_rule") != "lower bound strictly greater than 0.50"
    ):
        raise ValueError("registered_serving_gate_bootstrap_or_threshold_mismatch")

    authorization = registration.get("authorization", {})
    if authorization.get("currently_authorized") is not False:
        raise ValueError("registration_authorized_or_invalid")


def verify_candidate_rows(
    rows: list[dict[str, Any]],
    eligible: dict[str, dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> None:
    if len(rows) != sum(seed615.PHASE_COUNTS.values()):
        raise ValueError("candidate_row_count_mismatch")
    if Counter(row["phase"] for row in rows) != Counter(seed615.PHASE_COUNTS):
        raise ValueError("candidate_phase_quotas_mismatch")
    canonical_ids = [row["canonical_identity"] for row in rows]
    float_ids = [row["float32_identity"] for row in rows]
    if len(set(canonical_ids)) != len(rows):
        raise ValueError("candidate_canonical_duplicate")
    if len(set(float_ids)) != len(rows):
        raise ValueError("candidate_float32_duplicate")
    for row in rows:
        authenticated = eligible.get(row["canonical_identity"])
        if authenticated is None:
            raise ValueError("candidate_provenance_not_eligible")
        if canonical_bytes(authenticated) != canonical_bytes(row):
            raise ValueError("candidate_target_or_provenance_changed")
        seed599.validate_target_record(row)
    seed599.validate_training_heldout_disjointness(rows, [], heldout)

    # Recreate the max-min group allocation independently from a rank-ordered
    # list of per-group capacities and compare row membership, not just counts.
    expected_ids: set[str] = set()
    for phase, quota in seed615.PHASE_COUNTS.items():
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in eligible.values():
            if row["phase"] == phase:
                buckets[(row["source"], row["opening_id"])].append(row)
        group_keys = sorted(
            buckets,
            key=lambda group: (
                rank("seed615-group", phase, f"{group[0]}:{group[1]}"),
                group,
            ),
        )
        for group, group_rows in buckets.items():
            group_rows.sort(
                key=lambda row: (
                    rank("seed615-state", phase, row["canonical_identity"]),
                    row["canonical_identity"],
                )
            )
        chosen = 0
        round_number = 0
        while chosen < quota:
            progressed = False
            for group in group_keys:
                if round_number < len(buckets[group]) and chosen < quota:
                    expected_ids.add(buckets[group][round_number]["canonical_identity"])
                    chosen += 1
                    progressed = True
            if not progressed:
                raise ValueError("eligible_capacity_cannot_fill_quota")
            round_number += 1
    if set(canonical_ids) != expected_ids:
        raise ValueError("candidate_selection_algorithm_mismatch")


def verify_openings(
    rows: list[dict[str, Any]],
    authenticated: dict[str, Any],
    eligible: dict[str, dict[str, Any]],
) -> None:
    if len(rows) != seed615.EVALUATION_OPENING_COUNT:
        raise ValueError("evaluation_opening_count_mismatch")
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    heldout = authenticated["heldout"]
    blocked_canonical = (
        set(eligible)
        | {row["canonical_identity"] for row in heldout}
        | authenticated["prior_evaluation_opening_canonical"]
    )
    blocked_float = (
        {row["float32_identity"] for row in eligible.values()}
        | {row["float32_identity"] for row in heldout}
        | authenticated["prior_evaluation_opening_float32"]
    )
    for index, row in enumerate(rows):
        if row["opening_id"] != f"seed615-opening-{index:04d}":
            raise ValueError("evaluation_opening_id_mismatch")
        state = validate_opening_prefix(row["prefix_moves"])
        canonical = seed599.canonical_identity(state)
        float32 = seed599.production_float32_identity(state)
        if canonical != row["canonical_identity"] or canonical_bytes(
            state
        ) != canonical_bytes(row["state"]):
            raise ValueError("evaluation_opening_canonical_identity_mismatch")
        if float32 != row["float32_identity"]:
            raise ValueError("evaluation_opening_float32_identity_mismatch")
        if canonical in blocked_canonical or float32 in blocked_float:
            raise ValueError("evaluation_opening_overlaps_excluded_population")
        if canonical in canonical_seen or float32 in float_seen:
            raise ValueError("evaluation_opening_duplicate")
        canonical_seen.add(canonical)
        float_seen.add(float32)


def verify_assignments(
    assignments: list[dict[str, Any]], openings: list[dict[str, Any]]
) -> None:
    """Check every frozen game seed and seat assignment without playing games."""
    if len(assignments) != len(openings) * 4:
        raise ValueError("evaluation_assignment_count_mismatch")
    expected = []
    for index, opening in enumerate(openings):
        for candidate in ("B512", "S615"):
            for seat in (0, 1):
                digest = hashlib.sha256(
                    f"seed615\0{opening['opening_id']}\0{candidate}\0{seat}".encode()
                ).digest()
                expected.append(
                    {
                        "game_id": f"seed615-{candidate}-{index:04d}-p{seat}",
                        "game_seed": int.from_bytes(digest[:8], "big") % (2**31),
                        "opening_id": opening["opening_id"],
                        "opening_index": index,
                        "opening_identity": opening["canonical_identity"],
                        "candidate": candidate,
                        "candidate_seat": seat,
                        "opponent": "P-E4",
                    }
                )
    if canonical_bytes(assignments) != canonical_bytes(expected):
        raise ValueError("evaluation_assignment_contract_mismatch")


def verify(root: Path = ROOT) -> dict[str, Any]:
    authenticated = seed615.authenticated_population(root)
    destination = root / "docs/data/seed615-source-opening-diversity"
    candidate = read_jsonl(destination / "candidate-membership.jsonl")
    openings = read_jsonl(destination / "evaluation-openings.jsonl")
    assignments = read_jsonl(destination / "evaluation-assignments.jsonl")
    verify_candidate_rows(
        candidate, authenticated["eligible"], authenticated["heldout"]
    )
    verify_openings(openings, authenticated, authenticated["eligible"])
    verify_assignments(assignments, openings)
    membership_digest = hashlib.sha256(canonical_bytes(candidate)).hexdigest()
    opening_digest = hashlib.sha256(canonical_bytes(openings)).hexdigest()
    assignment_digest = hashlib.sha256(canonical_bytes(assignments)).hexdigest()
    if (
        membership_digest
        != "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010"
    ):
        raise ValueError("candidate_membership_digest_mismatch")
    if (
        opening_digest
        != "9bc2bbfca07ea356343c25e94593ab638b16238895fce5062f904d2cc2092cc0"
    ):
        raise ValueError("evaluation_opening_digest_mismatch")
    if (
        assignment_digest
        != "8fb71d6ee30ad6870d403843dbe1a4e798f0218617cee5fce21964372a8514ca"
    ):
        raise ValueError("evaluation_assignment_digest_mismatch")
    registration = json.loads(
        (
            root / "docs/data/seed615-source-opening-diversity/registration.json"
        ).read_bytes()
    )
    validate_registration_contract(
        registration,
        membership_digest=membership_digest,
        opening_digest=opening_digest,
        assignment_digest=assignment_digest,
    )
    return {
        "passed": True,
        "training_or_evaluation_started": False,
        "candidate_states": len(candidate),
        "candidate_membership_sha256": membership_digest,
        "evaluation_openings": len(openings),
        "evaluation_opening_manifest_sha256": opening_digest,
        "evaluation_assignments": len(assignments),
        "evaluation_assignment_manifest_sha256": assignment_digest,
        "registration_authenticated": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
