"""Read-only Seed 621 audit of authenticated #557/#559 roots and model priors.

This module never constructs a search or oracle. It validates the published
oracle joins and performs deterministic single-network inference only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import canonical_hash
from ml.alphazero_lite.seed615_policy_divergence_diagnostic import compare_priors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
SEED557 = DATA / "seed557-retrospective-diagnostic"
SEED559 = DATA / "seed559-fixed-position-feasibility/run-0003"
SEED615 = DATA / "seed615-source-opening-diversity"
FREEZE_SHA256 = "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
MODEL_HASHES = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
B512_MEMBERSHIP_SHA256 = (
    "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516"
)
HELDOUT_SHA256 = "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _states() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    source_rows: list[dict] = []
    oracle_by_hash: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for cohort_name, cohort_path, oracle_path in (
        ("seed557", SEED557 / "cohort.json", SEED557 / "oracle-records.jsonl"),
        ("seed559", SEED559 / "cohort-manifest.json", SEED559 / "oracle-records.jsonl"),
    ):
        states = json.loads(cohort_path.read_text())["states"]
        oracle_rows = jsonl(oracle_path)
        if len(oracle_rows) != len(states):
            raise ValueError(f"{cohort_name}_oracle_count_mismatch")
        cohort_map = {row["state_hash"]: row for row in states}
        if len(cohort_map) != len(states):
            raise ValueError(f"{cohort_name}_duplicate_state_hash")
        records = {row["state_hash"]: row for row in oracle_rows}
        if len(records) != len(oracle_rows) or set(records) != set(cohort_map):
            raise ValueError(f"{cohort_name}_oracle_identity_mismatch")
        for state_id, state_row in cohort_map.items():
            state = state_row["state"]
            game = KalahGame.from_state(state)
            legal = game.possible_moves()
            if canonical_hash(state) != state_id:
                raise ValueError(
                    f"{cohort_name}_canonical_identity_mismatch:{state_id}"
                )
            oracle = records[state_id]
            margins = {int(a): int(v) for a, v in oracle["action_margins"].items()}
            if oracle.get("coverage_complete") is not True or set(margins) != set(
                legal
            ):
                raise ValueError(f"{cohort_name}_legal_coverage_mismatch:{state_id}")
            if sorted(oracle["legal_moves"]) != sorted(legal):
                raise ValueError(f"{cohort_name}_legal_list_mismatch:{state_id}")
            if int(state["current_player"]) != game.current_player:
                raise ValueError(
                    f"{cohort_name}_player_orientation_mismatch:{state_id}"
                )
            # Published root margins are from the side-to-move perspective. A
            # complete action map and exact max are required; terminal states
            # and unreachable/synthetic rows are excluded by KalahGame.
            source_rows.append(state_row)
            oracle_by_hash.setdefault(state_id, oracle)
            provenance.setdefault(state_id, {"cohorts": [], "sources": []})
            provenance[state_id]["cohorts"].append(cohort_name)
            provenance[state_id]["sources"].append(
                {
                    "cohort": cohort_name,
                    "opening_index": state_row.get("opening_index"),
                    "source_path": state_row.get("source_path"),
                    "source_line": state_row.get("source_line"),
                    "source_record_sha256": state_row.get("source_record_sha256"),
                    "source_batch_sha256": state_row.get("source_batch_sha256"),
                }
            )
    return source_rows, oracle_by_hash, provenance


def _load_membership() -> tuple[dict[str, set[str]], dict[str, str]]:
    paths = {
        "B512": DATA / "seed599-replay-coverage/treatment-membership.json",
        "S615": SEED615 / "candidate-membership.jsonl",
        "protected_holdout": DATA / "seed599-replay-coverage/heldout-membership.json",
    }
    freeze = json.loads(
        (SEED615 / "evaluation-preparation/execution-freeze-v4.json").read_text()
    )
    if (
        file_sha(SEED615 / "evaluation-preparation/execution-freeze-v4.json")
        != FREEZE_SHA256
    ):
        raise ValueError("seed615_evaluation_freeze_hash_mismatch")
    declared = {
        "S615": freeze["registration"]["opening_verification"][
            "candidate_membership_sha256"
        ],
        "B512": B512_MEMBERSHIP_SHA256,
        "protected_holdout": HELDOUT_SHA256,
    }
    result: dict[str, set[str]] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        hashes[name] = sha(raw)
        rows = (
            [json.loads(line) for line in raw.splitlines() if line]
            if path.suffix == ".jsonl"
            else json.loads(raw)
        )
        if name == "S615":
            canonical = json.dumps(
                rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
            hashes[name] = sha(canonical)
        elif declared[name] is not None:
            canonical = json.dumps(
                rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
            hashes[name] = sha(canonical)
        if hashes[name] != declared[name]:
            raise ValueError(f"{name}_membership_hash_mismatch")
        identities = set()
        for row in rows:
            state = row.get("canonical_state")
            identity = row.get("canonical_identity")
            if state is not None and canonical_hash(state) != identity:
                raise ValueError(f"{name}_membership_state_hash_mismatch")
            if identity:
                identities.add(identity)
        result[name] = identities
    return result, hashes


def audit() -> dict[str, Any]:
    states, oracle, provenance = _states()
    by_id: dict[str, dict] = {}
    for row in states:
        by_id.setdefault(row["state_hash"], row)
    membership, membership_hashes = _load_membership()
    artifacts = SEED615 / "evaluation-preparation/runtime-artifacts"
    checkpoint_freeze = json.loads(
        (SEED615 / "evaluation-preparation/execution-freeze-v4.json").read_text()
    )
    models = {}
    runtime_hashes: dict[str, dict[str, str]] = {}
    for name, digest in MODEL_HASHES.items():
        directory = artifacts / name
        observed = {
            "model.npz": file_sha(directory / "model.npz"),
            "metadata.json": file_sha(directory / "metadata.json"),
            "weights.json": file_sha(directory / "weights.json"),
        }
        expected = checkpoint_freeze["runtime_artifacts"][name]
        for file_name, expected_field in (
            ("model.npz", "model_npz_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("weights.json", "weights_json_sha256"),
        ):
            if observed[file_name] != expected[expected_field]:
                raise ValueError(f"{name}_{file_name.replace('.', '_')}_hash_mismatch")
        if observed["model.npz"] != digest:
            raise ValueError(f"{name}_checkpoint_hash_mismatch")
        runtime_hashes[name] = observed
        models[name] = ArtifactEvaluator(directory)

    inventory = []
    reasons = Counter()
    cohort_metrics: dict[str, list[dict]] = defaultdict(list)
    for state_id, row in sorted(by_id.items()):
        reason = []
        if state_id in membership["B512"]:
            reason.append("B512_training_state_overlap")
        if state_id in membership["S615"]:
            reason.append("S615_training_state_overlap")
        if state_id in membership["protected_holdout"]:
            reason.append("protected_holdout_overlap")
        game = KalahGame.from_state(row["state"])
        b, _ = models["B512"].evaluate(game)
        s, _ = models["S615"].evaluate(game)
        comparison = compare_priors(b, s, game.possible_moves())
        record = {
            "state_id": state_id,
            "state": row["state"],
            "player_to_move": game.current_player,
            "legal_moves": game.possible_moves(),
            "exact_action_margins_to_move_perspective": oracle[state_id][
                "action_margins"
            ],
            "cohorts": provenance[state_id]["cohorts"],
            "source_clusters": [
                {
                    "opening_index": x.get("opening_index"),
                    "source_path": x.get("source_path"),
                    "source_line": x.get("source_line"),
                    "source_record_sha256": x.get("source_record_sha256"),
                    "source_batch_sha256": x.get("source_batch_sha256"),
                }
                for x in provenance[state_id]["sources"]
            ],
            "eligible_by_exact_membership_rule": not reason,
            "exclusion_reasons": reason,
            "contamination_status": "excluded_exact_membership"
            if reason
            else "unresolved_nonmembership_not_proven",
            **comparison,
        }
        inventory.append(record)
        if not reason:
            for cohort in record["cohorts"]:
                cohort_metrics[cohort].append(record)
        for item in reason:
            reasons[item] += 1
    eligible = [row for row in inventory if row["eligible_by_exact_membership_rule"]]
    strata = Counter(
        "agreement" if row["top_prior_agreement"] else "disagreement"
        for row in eligible
    )
    summary = {
        "schema": "seed621-oracle-cohort-audit-v1",
        "read_only": True,
        "source_commits": {
            "seed557_and_seed559_reconstruction": "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae",
        },
        "source_artifacts_sha256": {
            str(path.relative_to(ROOT)): file_sha(path)
            for path in (
                SEED557 / "cohort.json",
                SEED557 / "oracle-records.jsonl",
                SEED557 / "execution-freeze.json",
                SEED557 / "oracle-identity-verification.json",
                SEED559 / "cohort-manifest.json",
                SEED559 / "oracle-records.jsonl",
                SEED559 / "execution-freeze.json",
                DATA
                / "seed556-exploratory-arena/screen-0005/verification-receipt.json",
                SEED615 / "training-run-S615/training-receipt.json",
                SEED615 / "training-run-S615/training-integrity-report.json",
                SEED615 / "evaluation-preparation/execution-freeze-v4.json",
            )
        },
        "checkpoint_sha256": MODEL_HASHES,
        "runtime_artifact_sha256": runtime_hashes,
        "membership_sha256": membership_hashes,
        "counts": {
            "source_records": len(states),
            "unique_states": len(by_id),
            "duplicated_across_cohorts": len(states) - len(by_id),
            "exact_covered": len(by_id),
            "exact_covered_excluded_by_authenticated_membership": len(inventory)
            - len(eligible),
            "eligible_by_exact-membership-rule": len(eligible),
            "unresolved_nonmembership": len(eligible),
            "agreement": strata["agreement"],
            "disagreement": strata["disagreement"],
            "exclusion_reasons": dict(reasons),
        },
        "cohort_policy_metrics": {
            cohort: {
                "n": len(rows),
                "agreements": sum(r["top_prior_agreement"] for r in rows),
                "disagreements": sum(not r["top_prior_agreement"] for r in rows),
                "mean_tv": float(np.mean([r["total_variation"] for r in rows]))
                if rows
                else None,
                "mean_js_bits": float(np.mean([r["jensen_shannon_bits"] for r in rows]))
                if rows
                else None,
            }
            for cohort, rows in cohort_metrics.items()
        },
        "decision_gate": {
            "minimum_64_and_16_each_class_met": len(eligible) >= 64
            and strata["agreement"] >= 16
            and strata["disagreement"] >= 16,
            "training_contamination_excluded": False,
            "recommendation": "revise",
        },
        "inventory": inventory,
    }
    return summary


def verify_report(report: dict[str, Any]) -> None:
    """Independent structural assertions used by verifier and negative tests."""
    rows = report["inventory"]
    ids = [row["state_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_inventory_state")
    for row in rows:
        game = KalahGame.from_state(row["state"])
        if canonical_hash(row["state"]) != row["state_id"]:
            raise ValueError("state_identity_mismatch")
        if set(map(int, row["exact_action_margins_to_move_perspective"])) != set(
            game.possible_moves()
        ):
            raise ValueError("oracle_coverage_mismatch")
        if (
            row["eligible_by_exact_membership_rule"]
            and row["contamination_status"] != "unresolved_nonmembership_not_proven"
        ):
            raise ValueError("eligible_contamination_must_be_explicitly_unresolved")
        if row["eligible_by_exact_membership_rule"] and row["exclusion_reasons"]:
            raise ValueError("eligible_state_has_exclusion_reason")


if __name__ == "__main__":
    result = audit()
    verify_report(result)
    destination = ROOT / "docs/data/seed621-oracle-cohort-audit"
    destination.mkdir(parents=True, exist_ok=True)
    inventory_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in result.pop("inventory")
    )
    result["inventory_sha256"] = sha(inventory_bytes.encode())
    for name, payload in (
        ("eligible-state-inventory.jsonl", inventory_bytes),
        ("report.json", json.dumps(result, sort_keys=True, indent=2) + "\n"),
    ):
        target = destination / name
        target.write_text(payload)
    print(json.dumps(result["counts"], sort_keys=True, indent=2))
