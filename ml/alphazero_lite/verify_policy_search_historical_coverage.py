"""Independently verify the frozen retrospective evidence-coverage join."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED615 = Path("docs/data/seed615-source-opening-diversity/posthoc-policy-diagnostic")
SEED633 = Path("docs/data/seed633-exact-oracle-decision-diagnostic/attempt-0003")
OUTPUT = Path("docs/data/seed633-policy-search-retrospective/coverage.json")
B512_SHA256 = "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"
S615_SHA256 = "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780"

ANCHORS = {
    SEED615
    / "diagnostic.json": "bb5d8099950b16d3e8cbc0b0e9f2b9504e43949e7061aa0b5ab13357bc75f29a",
    SEED615
    / "first-divergences.jsonl": "05cc496f17c530727aac05173dfcd508385800bddb3ed3dce9357528dcb3a72c",
    SEED615
    / "paired-opening-outcomes.jsonl": "30cca94e5a84aea64219806b84b94b8965830c3da0c591321ef386b2c6ba9973",
    SEED615
    / "policy-prior-comparison.jsonl": "00857989a6d4d86c6f6009669a54ec9776f441a665c36be7f3faa878221c9fb4",
    SEED633
    / "freeze.json": "21fe37edfb691a5a707a966ef07903675e42c790881f195ee4a29e2281b312fc",
    SEED633
    / "per-state-results.jsonl": "ead34f5772ec3880856efdddcf7597162f2b62a3b26a71dfe6a0424850003815",
    Path(
        "docs/data/seed633-teacher-target-learning/attempt-0005/cohort.jsonl"
    ): "303e6c38feff9868133e84a80b92421e26da480dbea36dd3a37a8f70d33e914d",
    Path(
        "docs/data/seed633-teacher-target-learning/attempt-0005/per-state-metrics.jsonl"
    ): "ebba7af50e154ff6e31e3afce3ee325da915295ac11f8a10a73f7a516dd2cccb",
    Path(
        "docs/data/seed633-teacher-target-learning/attempt-0005/summary.json"
    ): "b9301d5bfcb7d80cdea8b311247637a8ef21c33820b0eedfe3776a2d3b2ccf29",
    Path(
        "docs/data/seed633-teacher-target-learning/attempt-0005/independent-verification.json"
    ): "34aa59dea2547c81aba439aeaef2fb8e8dc56d307f2dbda686d2fb2b45aba639",
}


def sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def verify_bytes_identity(path: Path, payload: bytes, expected: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"historical_source_identity_mismatch:{path}")


def verify_anchors(anchors: dict[Path, str] = ANCHORS) -> None:
    for path, expected in anchors.items():
        verify_bytes_identity(path, (ROOT / path).read_bytes(), expected)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def index_unique(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get(field)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"missing_canonical_identity:{field}")
        if identity in indexed:
            raise ValueError(f"repeated_identity_requires_occurrence_review:{identity}")
        indexed[identity] = row
    return indexed


def validate_search_pair(row: dict[str, Any]) -> None:
    for key in ("b512_move", "s615_move", "player_to_move", "b512_seed", "s615_seed"):
        if not isinstance(row.get(key), int):
            raise ValueError(f"missing_recorded_search_evidence:{key}")
    if row["b512_seed"] == row["s615_seed"]:
        raise ValueError("candidate_search_seeds_not_distinct")
    state = row.get("state_before_divergence")
    if not isinstance(state, dict):
        raise ValueError("missing_state_for_perspective_check")
    if state.get("current_player") != row["player_to_move"]:
        raise ValueError("player_perspective_mismatch")
    policy = row.get("policy_prior_comparison")
    if not isinstance(policy, dict):
        raise ValueError("missing_raw_policy_comparison")
    if policy.get("canonical_state_identity") != row.get("canonical_state_identity"):
        raise ValueError("policy_state_identity_mismatch")


def build_coverage() -> dict[str, Any]:
    verify_anchors()
    policies = read_jsonl(SEED615 / "policy-prior-comparison.jsonl")
    exact = read_jsonl(SEED633 / "per-state-results.jsonl")
    divergences = read_jsonl(SEED615 / "first-divergences.jsonl")
    exact_freeze = json.loads((ROOT / SEED633 / "freeze.json").read_text())
    if exact_freeze.get("checkpoint_sha256") != {
        "B512": B512_SHA256,
        "S615": S615_SHA256,
    }:
        raise ValueError("exact_source_checkpoint_identity_mismatch")
    for row in policies:
        provenance = row.get("provenance", {})
        if (
            provenance.get("B512_checkpoint_sha256") != B512_SHA256
            or provenance.get("S615_checkpoint_sha256") != S615_SHA256
        ):
            raise ValueError("policy_source_checkpoint_identity_mismatch")
    policy_by_id = index_unique(policies, "canonical_state_identity")
    exact_by_id = index_unique(exact, "canonical_identity")
    divergence_by_id = {
        row["canonical_state_identity"]: row
        for row in divergences
        if isinstance(row.get("canonical_state_identity"), str)
    }

    overlap = sorted(policy_by_id.keys() & exact_by_id.keys())
    by_membership = Counter(exact_by_id[key]["membership_class"] for key in overlap)
    by_phase = Counter(exact_by_id[key]["phase"] for key in overlap)
    by_membership_phase = Counter(
        (exact_by_id[key]["membership_class"], exact_by_id[key]["phase"])
        for key in overlap
    )
    actionable: list[dict[str, Any]] = []
    for identity in overlap:
        row = divergence_by_id.get(identity)
        if row is None:
            continue
        provenance = row.get("provenance", {})
        if (
            provenance.get("B512_checkpoint_sha256") != B512_SHA256
            or provenance.get("S615_checkpoint_sha256") != S615_SHA256
        ):
            raise ValueError("recorded_action_checkpoint_identity_mismatch")
        validate_search_pair(row)
        actionable.append(row)

    exact_membership_totals = Counter(row["membership_class"] for row in exact)
    exact_phase_totals = Counter(row["phase"] for row in exact)
    represented_memberships = len(by_membership)
    represented_phases = len(by_phase)
    exact_margin_rows = sum(
        bool(row.get("exact_action_margins")) for row in exact_by_id.values()
    )
    result = {
        "schema": "policy-to-search-historical-coverage-v1",
        "evidence_classification": "authenticated_retrospective_coverage_only",
        "sources": {
            "seed615_unique_policy_states": len(policy_by_id),
            "seed615_first_divergence_occurrences": len(divergences),
            "seed633_unique_exact_oracle_states": len(exact_by_id),
            "seed633_states_with_exact_margins": exact_margin_rows,
        },
        "matched": {
            "unique_exact_identity_states": len(overlap),
            "matched_decisions_with_both_raw_policies": len(overlap),
            "matched_decisions_with_exact_action_margins": sum(
                bool(exact_by_id[key].get("exact_action_margins")) for key in overlap
            ),
            "matched_decisions_with_both_recorded_search_actions": len(actionable),
            "distinct_seed615_opening_groups": len(
                {row["opening_index"] for row in actionable}
            ),
            "by_membership_class": dict(sorted(by_membership.items())),
            "by_phase": dict(sorted(by_phase.items())),
            "by_membership_class_and_phase": {
                f"{membership}|{phase}": count
                for (membership, phase), count in sorted(by_membership_phase.items())
            },
        },
        "seed633_exact_cohort_coverage": {
            "by_membership_class": dict(sorted(exact_membership_totals.items())),
            "by_phase": dict(sorted(exact_phase_totals.items())),
        },
        "go_no_go": {
            "minimum_unique_matched_roots": 100,
            "minimum_exact_margin_roots": 50,
            "minimum_opening_groups": 20,
            "minimum_membership_classes": 2,
            "minimum_phase_strata": 1,
            "observed_membership_classes": represented_memberships,
            "observed_phase_strata": represented_phases,
            "passed": (
                len(overlap) >= 100
                and sum(
                    bool(exact_by_id[key].get("exact_action_margins"))
                    for key in overlap
                )
                >= 50
                and len({row["opening_index"] for row in actionable}) >= 20
                and represented_memberships >= 2
                and represented_phases >= 1
            ),
            "reason": "matched historical coverage falls below the frozen minimums",
        },
        "limitations": [
            "Seed 615 game seeds are candidate-specific; paired outcomes and action differences are descriptive, not causal.",
            "The source-opening-label to Seed 624 opening-identifier crosswalk remains unresolved.",
            "No action-quality contrasts or mechanism classifications were computed after the failed gate.",
            "No new inference, MCTS, game generation, solving, or tablebase probing was performed.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    coverage = build_coverage()
    output = ROOT / args.output
    if args.verify_existing:
        if not output.exists():
            raise FileNotFoundError(f"coverage_report_missing:{args.output}")
        if json.loads(output.read_text()) != coverage:
            raise ValueError("published_coverage_report_mismatch")
        print(
            json.dumps(
                {"status": "verified", "coverage": coverage["matched"]}, sort_keys=True
            )
        )
        return
    if output.exists():
        raise FileExistsError(f"create_only_output_exists:{args.output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps(coverage["matched"], sort_keys=True))


if __name__ == "__main__":
    main()
