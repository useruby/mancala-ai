"""Relocatable, read-only verifier for the seed558 paired value swap."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import arena
from ml.alphazero_lite.seed558_value_only_swap import ARMS, DATA, score

EXPECTED_D432 = "4555cba73d2c908d1eca829836f361b226ccc1072be28affd816b2a240123163"
EXPECTED_COHORT = "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243"
EXPECTED_ORACLES = "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480"
EXPECTED_PUBLICATIONS = {
    "seed546_merge_commit": "54ca41c2f499452a424dfdd32f0aede1b584fcb7",
    "seed546_checkpoint_sha256": EXPECTED_D432,
    "seed546_registration_sha256": "8541fef485ac915cbf2b32ac7b46f49e43e9fa867fb8411822594eaa824059f6",
    "seed546_execution_freeze_sha256": "65286df075d662232a439e7288285fd31d34a9cfc5f4d5d5aa2b6b4006cd60ab",
    "seed546_runtime_authentication_sha256": "570b4c71f37999aa4ba7d4872122276d2556f7a7763bfb14fb3a9921ec1dba53",
    "seed557_merge_commit": "44834acaa0d7ee17465e807d10bc90e841a51a89",
    "seed557_execution_freeze_sha256": "b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d",
    "seed557_oracle_identity_record_sha256": "45f93ce95bf2bc0c58b8a44bdf072324eddfb2c943dd9678e6895613a5a4f60d",
    "serving_runtime_authentication_sha256": "61372a7581743be18b7eeebf8922da4ecc0a2a8e440113a80103fea9b417be04",
}
EXPECTED_557_RECEIPTS = {
    "seed555_verification_receipt_sha256": "a12b3284acb72ab817e27e79a265097da3873e7570c93fdbcd3a47810028dc37",
    "seed556_verification_receipt_sha256": "a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify(data: Path = DATA) -> dict[str, Any]:
    """Validate evidence consistency without consulting source checkout paths."""
    freeze = json.loads((data / "execution-freeze.json").read_text())
    cohort_bytes = (data / "cohort.json").read_bytes()
    oracle_bytes = (data / "oracle-records.jsonl").read_bytes()
    if freeze["schema"] != "seed558-value-only-freeze-v1":
        raise ValueError("freeze_schema_mismatch")
    for name, value in EXPECTED_PUBLICATIONS.items():
        if freeze["publications"].get(name) != value:
            raise ValueError(f"published_source_identity_mismatch:{name}")
    if (
        freeze["publications"].get("seed557_policy_publication_receipts")
        != EXPECTED_557_RECEIPTS
    ):
        raise ValueError("seed557_policy_receipt_binding_mismatch")
    if (
        _sha(cohort_bytes) != freeze["cohort_sha256"]
        or _sha(oracle_bytes) != freeze["oracle_records_sha256"]
    ):
        raise ValueError("frozen_cohort_or_oracle_hash_mismatch")
    if (
        freeze["cohort_sha256"] != EXPECTED_COHORT
        or freeze["oracle_records_sha256"] != EXPECTED_ORACLES
    ):
        raise ValueError("source_557_cohort_or_oracle_identity_mismatch")
    cohort_doc = json.loads(cohort_bytes)
    cohort = cohort_doc["states"]
    oracles = [json.loads(line) for line in oracle_bytes.splitlines() if line]
    if len(cohort) != 256 or len(oracles) != 256:
        raise ValueError("frozen_coverage_mismatch")
    state_by_hash = {row["state_hash"]: row for row in cohort}
    if len(state_by_hash) != 256 or {r["state_hash"] for r in oracles} != set(
        state_by_hash
    ):
        raise ValueError("duplicate_or_missing_frozen_root")
    for arm, artifacts in freeze["components"].items():
        for filename, digest in artifacts.items():
            actual = _sha((data / "artifacts" / arm / filename).read_bytes())
            if actual != digest:
                raise ValueError(f"component_artifact_hash_mismatch:{arm}/{filename}")
    if freeze["components"]["D432"]["model.npz"] != EXPECTED_D432:
        raise ValueError("D432_artifact_identity_mismatch")
    if (
        freeze["components"]["P_E4"]["model.npz"]
        != "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
    ):
        raise ValueError("serving_P_artifact_identity_mismatch")
    if (
        freeze["components"]["initializer"]["model.npz"]
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise ValueError("initializer_artifact_identity_mismatch")
    p = arena.ArtifactEvaluator(data / "artifacts/P_E4")
    initializer = arena.ArtifactEvaluator(data / "artifacts/initializer")
    d432 = arena.ArtifactEvaluator(data / "artifacts/D432")
    compositions = {
        "initializer_value": arena.ComposedArtifactEvaluator(
            p, initializer, policy_source="current", value_source="candidate"
        ),
        "D432_value": arena.ComposedArtifactEvaluator(
            p, d432, policy_source="current", value_source="candidate"
        ),
    }
    oracle_by_hash = {r["state_hash"]: r for r in oracles}
    for state_hash, item in state_by_hash.items():
        legal = KalahGame.from_state(item["state"]).possible_moves()
        oracle = oracle_by_hash[state_hash]
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        if oracle.get("coverage_complete") is not True or set(margins) != set(legal):
            raise ValueError("oracle_legal_action_coverage_mismatch")
        if sorted(oracle["legal_moves"]) != sorted(legal):
            raise ValueError("oracle_legal_moves_mismatch")
    searches = _jsonl(data / "search-records.jsonl")
    if len(searches) != 512:
        raise ValueError("search_record_count_mismatch")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in searches:
        key = (row["state_hash"], row["arm"])
        if (
            row["arm"] not in ARMS
            or key in by_key
            or row["state_hash"] not in state_by_hash
        ):
            raise ValueError("duplicate_or_unknown_search_record")
        expected_seed = int(_sha(f"seed558:{row['state_hash']}".encode())[:8], 16) % (
            2**31
        )
        if int(row["seed"]) != expected_seed:
            raise ValueError("matched_seed_does_not_follow_frozen_contract")
        by_key[key] = row
        legal = KalahGame.from_state(
            state_by_hash[row["state_hash"]]["state"]
        ).possible_moves()
        if sorted(row["legal_moves"]) != sorted(legal):
            raise ValueError("search_legal_moves_mismatch")
        state = KalahGame.from_state(state_by_hash[row["state_hash"]]["state"])
        recomputed_priors, _ = compositions[row["arm"]].evaluate(state)
        if row["priors"] != [float(value) for value in recomputed_priors]:
            raise ValueError("recorded_priors_do_not_match_authenticated_policy")
        if len(row["priors"]) != 6 or any(
            not math.isfinite(float(p)) or float(p) < 0 for p in row["priors"]
        ):
            raise ValueError("invalid_policy_priors")
        visits = [float(v) for v in row["visits"]]
        if len(visits) != 6 or any(not v.is_integer() or v < 0 for v in visits):
            raise ValueError("invalid_visit_vector")
        if sum(visits) != 128 or any(visits[i] for i in range(6) if i not in legal):
            raise ValueError("simulation_budget_or_illegal_visit_mismatch")
        max_visits = max(visits[move] for move in legal)
        winners = [move for move in legal if visits[move] == max_visits]
        if row["selected_action"] not in winners:
            raise ValueError("selected_action_contradicts_visits")
        stats = {int(s["move"]): s for s in row["child_stats"]}
        if set(stats) != set(legal):
            raise ValueError("child_Q_coverage_mismatch")
        if any(
            int(stats[m]["visits"]) != int(visits[m])
            or not math.isfinite(float(stats[m]["q_value"]))
            for m in legal
        ):
            raise ValueError("child_Q_or_visit_binding_mismatch")
        expected_action = max(
            winners,
            key=lambda m: (
                visits[m],
                float(stats[m]["q_value"]),
                float(row["priors"][m]),
                -m,
            ),
        )
        if row["selected_action"] != expected_action:
            raise ValueError("selected_action_breaks_deterministic_tie_rule")
    for state_hash in state_by_hash:
        left, right = (by_key[(state_hash, arm)] for arm in ARMS)
        if left["seed"] != right["seed"]:
            raise ValueError("matched_seed_mismatch")
        if left["priors"] != right["priors"]:
            raise ValueError("policy_priors_changed_between_arms")
    metrics = score(cohort, searches, oracles)
    roots = metrics.pop("root_records")
    published_roots = _jsonl(data / "root-metrics.jsonl")
    if roots != published_roots:
        raise ValueError("root_metric_reconstruction_mismatch")
    published = json.loads((data / "analysis.json").read_text())
    if metrics != published:
        raise ValueError("analysis_reconstruction_mismatch")
    expected_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
    }
    contract = freeze["search"]
    if (
        contract["simulations"] != 128
        or contract["c_puct"] != 1.25
        or contract["workers"] != 1
        or contract["fresh_tree_per_root_arm"] is not True
        or contract["same_seed_per_root"] is not True
        or contract["exact_root_solving"] is not False
        or contract["exact_leaf_solving"] is not False
        or contract["search_options"] != expected_options
    ):
        raise ValueError("search_contract_mismatch")
    return {
        "status": "verified",
        "roots": 256,
        "paired_searches": 512,
        "oracle_roots": 256,
        "decision": published["decision"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
