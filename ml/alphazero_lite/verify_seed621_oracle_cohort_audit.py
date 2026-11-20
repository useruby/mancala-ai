"""Independent structural verifier for the published Seed 621 audit package."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed621-oracle-cohort-audit"
MEMBERSHIP_EVIDENCE = {
    "B512": (
        ROOT / "docs/data/seed599-replay-coverage/treatment-membership.json",
        "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
    ),
    "S615": (
        ROOT / "docs/data/seed615-source-opening-diversity/candidate-membership.jsonl",
        "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010",
    ),
    "protected_holdout": (
        ROOT / "docs/data/seed599-replay-coverage/heldout-membership.json",
        "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8",
    ),
}
CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha(rows: Any) -> str:
    return sha(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def verify() -> dict[str, Any]:
    report_path = OUT / "report.json"
    inventory_path = OUT / "eligible-state-inventory.jsonl"
    report = json.loads(report_path.read_bytes())
    payload = inventory_path.read_bytes()
    if sha(payload) != report["inventory_sha256"]:
        raise ValueError("inventory_digest_mismatch")
    inventory = [json.loads(line) for line in payload.splitlines() if line]
    ids = [row["state_id"] for row in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_state_inventory")
    strata = Counter()
    excluded = Counter()
    eligible_by_cohort: dict[str, Counter] = {}
    for row in inventory:
        state = row["state"]
        if canonical_hash(state) != row["state_id"]:
            raise ValueError("canonical_state_identity_mismatch")
        game = KalahGame.from_state(state)
        legal = set(game.possible_moves())
        margins = {
            int(key): int(value)
            for key, value in row["exact_action_margins_to_move_perspective"].items()
        }
        if legal != set(row["legal_moves"]) or legal != set(margins):
            raise ValueError("legal_action_oracle_coverage_mismatch")
        if row["player_to_move"] != game.current_player:
            raise ValueError("player_to_move_orientation_mismatch")
        if row["eligible_by_exact_membership_rule"]:
            if (
                row["exclusion_reasons"]
                or row["contamination_status"] != "unresolved_nonmembership_not_proven"
            ):
                raise ValueError("eligibility_or_contamination_label_mismatch")
            strata["agreement" if row["top_prior_agreement"] else "disagreement"] += 1
            for cohort in row["cohorts"]:
                eligible_by_cohort.setdefault(cohort, Counter())[
                    "agreement" if row["top_prior_agreement"] else "disagreement"
                ] += 1
        for reason in row["exclusion_reasons"]:
            excluded[reason] += 1
        for field in ("total_variation", "jensen_shannon_bits"):
            if float(row[field]) < 0:
                raise ValueError("negative_prior_divergence")
    counts = report["counts"]
    if (
        counts["unique_states"] != len(inventory)
        or counts["agreement"] != strata["agreement"]
        or counts["disagreement"] != strata["disagreement"]
    ):
        raise ValueError("published_count_mismatch")
    if counts["exact_covered_excluded_by_authenticated_membership"] != sum(
        excluded.values()
    ):
        raise ValueError("exclusion_count_mismatch")
    for relative, digest in report["source_artifacts_sha256"].items():
        if sha((ROOT / relative).read_bytes()) != digest:
            raise ValueError(f"source_artifact_digest_mismatch:{relative}")
    observed_memberships = {}
    for name, (path, expected) in MEMBERSHIP_EVIDENCE.items():
        raw = path.read_bytes()
        rows = (
            [json.loads(line) for line in raw.splitlines() if line]
            if path.suffix == ".jsonl"
            else json.loads(raw)
        )
        digest = canonical_sha(rows)
        if digest != expected:
            raise ValueError(f"membership_source_digest_mismatch:{name}")
        observed_memberships[name] = digest
    if report["membership_sha256"] != observed_memberships:
        raise ValueError("published_membership_hash_mismatch")
    for name, expected in CHECKPOINTS.items():
        directory = (
            ROOT
            / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
            / name
        )
        observed = {
            filename: sha((directory / filename).read_bytes())
            for filename in ("model.npz", "metadata.json", "weights.json")
        }
        if observed["model.npz"] != expected:
            raise ValueError(f"checkpoint_source_digest_mismatch:{name}")
        if observed != report["runtime_artifact_sha256"][name]:
            raise ValueError(f"runtime_artifact_report_digest_mismatch:{name}")
    if report["decision_gate"]["training_contamination_excluded"] is not False:
        raise ValueError("unresolved_training_contamination_must_fail_gate")
    return {
        "status": "verified",
        "unique_states": len(inventory),
        "top_prior_agreement": strata["agreement"],
        "top_prior_disagreement": strata["disagreement"],
        "contamination_clearance": "unresolved",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, indent=2))
