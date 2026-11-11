"""Read-only verifier for a #512 paired root-evaluation evidence file."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed512_policy_strength import (
    ARTIFACT_HASHES,
    ROOT,
    configure_evaluation_runtime,
    reconstruct_gate,
    sha256,
    verify_authorization,
    verify_freeze,
)
from ml.alphazero_lite import arena
from ml.alphazero_lite.seed557_retrospective import canonical_hash, search_seed


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"non_integer_evidence:{label}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid_numeric_evidence:{label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite_evidence:{label}")
    return result


def validate_paired_evidence(
    paired: Any,
    members: dict[str, dict[str, Any]],
    oracle_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate search-choice witnesses, then derive gate inputs from oracle."""
    if not isinstance(paired, dict) or set(paired) != {"control", "treatment"}:
        raise ValueError("evidence_arm_set_mismatch")
    if any(set(paired[arm]) != set(members) for arm in ("control", "treatment")):
        raise ValueError("evidence_membership_or_pair_incomplete")

    normalized: dict[str, dict[str, Any]] = {"control": {}, "treatment": {}}
    for arm in ("control", "treatment"):
        for key, member in members.items():
            row = paired[arm][key]
            oracle = oracle_rows.get(key)
            if oracle is None or oracle.get("coverage_complete") is not True:
                raise ValueError(f"oracle_coverage_missing:{key}")
            legal = [int(move) for move in member["legal_moves"]]
            oracle_legal = sorted(int(move) for move in oracle["legal_moves"])
            margins = {
                int(move): _exact_int(value, f"oracle_margin:{key}:{move}")
                for move, value in oracle["action_margins"].items()
            }
            if oracle_legal != legal or set(margins) != set(legal):
                raise ValueError(f"oracle_action_coverage_mismatch:{key}")

            opening_index = _exact_int(row.get("opening_index"), f"opening:{key}")
            seed = _exact_int(row.get("seed"), f"seed:{key}")
            expected_seed = int(search_seed(canonical_hash(member["state"])))
            if opening_index != int(member["opening_index"]) or seed != expected_seed:
                raise ValueError(f"root_pair_binding_mismatch:{arm}:{key}")

            action = _exact_int(row.get("selected_action"), f"action:{key}")
            if action not in legal:
                raise ValueError(f"selected_action_illegal:{arm}:{key}")
            witness = row.get("search_witness")
            if not isinstance(witness, dict):
                raise ValueError(f"search_witness_missing:{arm}:{key}")
            if (
                _exact_int(witness.get("selected_action"), f"witness_action:{key}")
                != action
            ):
                raise ValueError(f"search_selected_action_mismatch:{arm}:{key}")
            witness_legal = witness.get("legal_moves")
            if (
                not isinstance(witness_legal, list)
                or [
                    _exact_int(move, f"witness_legal_move:{key}")
                    for move in witness_legal
                ]
                != legal
            ):
                raise ValueError(f"search_witness_legal_moves_mismatch:{arm}:{key}")
            visit_counts = witness.get("visit_counts")
            priors = witness.get("network_priors")
            stats = witness.get("child_stats")
            if not isinstance(visit_counts, list) or len(visit_counts) != 6:
                raise ValueError(f"search_visit_vector_invalid:{arm}:{key}")
            visit_counts = [
                _exact_int(value, f"visits:{key}:{move}")
                for move, value in enumerate(visit_counts)
            ]
            if sum(visit_counts) != 128 or any(
                visit_counts[move] != 0 for move in range(6) if move not in legal
            ):
                raise ValueError(f"search_visit_budget_mismatch:{arm}:{key}")
            if not isinstance(priors, list) or len(priors) != 6:
                raise ValueError(f"search_prior_vector_invalid:{arm}:{key}")
            priors = [
                _finite_number(value, f"prior:{key}:{move}")
                for move, value in enumerate(priors)
            ]
            if any(priors[move] < 0 for move in legal):
                raise ValueError(f"negative_search_prior:{arm}:{key}")
            if not isinstance(stats, list) or len(stats) != len(legal):
                raise ValueError(f"search_child_stats_invalid:{arm}:{key}")
            stats_by_move = {}
            for stat in stats:
                if not isinstance(stat, dict):
                    raise ValueError(f"search_child_stats_invalid:{arm}:{key}")
                move = _exact_int(stat.get("move"), f"child_move:{key}")
                child_visits = _exact_int(
                    stat.get("visits"), f"child_visits:{key}:{move}"
                )
                q_value = _finite_number(stat.get("q_value"), f"child_q:{key}:{move}")
                if (
                    move in stats_by_move
                    or move not in legal
                    or child_visits != visit_counts[move]
                ):
                    raise ValueError(f"search_child_stats_mismatch:{arm}:{key}")
                stats_by_move[move] = {"visits": child_visits, "q_value": q_value}
            if set(stats_by_move) != set(legal):
                raise ValueError(f"search_child_coverage_mismatch:{arm}:{key}")
            search_selected = max(
                legal,
                key=lambda move: (
                    visit_counts[move],
                    stats_by_move[move]["q_value"],
                    priors[move],
                    -move,
                ),
            )
            if action != search_selected:
                raise ValueError(f"search_selected_action_mismatch:{arm}:{key}")

            margin = margins[action]
            best_margin = max(margins.values())
            regret = best_margin - margin
            wdl = (margin > 0) - (margin < 0)
            oracle_wdl = (best_margin > 0) - (best_margin < 0)
            for field, expected in (
                ("exact_final_margin", margin),
                ("regret", regret),
                ("wdl", wdl),
                ("oracle_wdl", oracle_wdl),
            ):
                if _exact_int(row.get(field), f"{field}:{key}") != expected:
                    raise ValueError(f"oracle_outcome_mismatch:{arm}:{key}:{field}")
            normalized[arm][key] = {
                "opening_index": opening_index,
                "seed": seed,
                "selected_action": action,
                "exact_final_margin": margin,
                "regret": regret,
                "wdl": wdl,
                "oracle_wdl": oracle_wdl,
            }

    for key in members:
        left, right = normalized["control"][key], normalized["treatment"][key]
        if (
            left["opening_index"] != right["opening_index"]
            or left["seed"] != right["seed"]
        ):
            raise ValueError(f"unmatched_pair:{key}")
    return normalized


def verify_search_replays(
    paired: dict[str, Any],
    members: dict[str, dict[str, Any]],
    replay_search: Callable[[str, str, dict[str, Any], int], dict[str, Any]],
) -> None:
    """Replay every paired frozen search and compare its complete root witness."""
    for arm in ("control", "treatment"):
        for key, member in members.items():
            recorded = paired[arm][key]["search_witness"]
            observed = replay_search(
                arm,
                key,
                member,
                _exact_int(paired[arm][key]["seed"], f"seed:{key}"),
            )
            if recorded != observed:
                raise ValueError(f"frozen_search_replay_mismatch:{arm}:{key}")


def verify_paired_payload(
    evidence: dict[str, Any],
    members: dict[str, dict[str, Any]],
    oracle_rows: dict[str, dict[str, Any]],
    replay_search: Callable[[str, str, dict[str, Any], int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if evidence.get("schema") != "seed512-policy-strength-evidence-v1":
        raise ValueError("evidence_schema_mismatch")
    normalized = validate_paired_evidence(evidence.get("pairs"), members, oracle_rows)
    verify_search_replays(evidence["pairs"], members, replay_search)
    observed_gate = reconstruct_gate(normalized)
    if evidence.get("gate") != observed_gate:
        raise ValueError("gate_reconstruction_mismatch")
    return normalized, observed_gate


def verify_evidence(
    root: Path,
    evidence: dict[str, Any],
    authorization: Path | None,
    trusted_authorization_sha256: str | None,
) -> dict[str, Any]:
    frozen = verify_freeze(root)
    if authorization is None or not authorization.is_file():
        raise PermissionError("post_review_authorization_missing")
    if trusted_authorization_sha256 is None:
        raise PermissionError("trusted_authorization_digest_missing")
    freeze_bytes = (
        root / "docs/data/seed512-policy-strength-evaluation/freeze.json"
    ).read_bytes()
    authorization_bytes = authorization.read_bytes()
    verify_authorization(
        freeze_bytes, authorization_bytes, trusted_authorization_sha256
    )
    authorization_sha256 = sha256(authorization_bytes)
    if evidence.get("schema") != "seed512-policy-strength-evidence-v1":
        raise ValueError("evidence_schema_mismatch")
    if (
        evidence.get("freeze_sha256") != sha256(freeze_bytes)
        or evidence.get("authorization_sha256") != authorization_sha256
        or evidence.get("membership_sha256")
        != frozen["freeze"]["cohort"]["membership_sha256"]
        or evidence.get("artifact_hashes") != ARTIFACT_HASHES
    ):
        raise ValueError("evidence_freeze_binding_mismatch")
    members = {row["state_hash"]: row for row in frozen["membership"]["roots"]}
    oracle_path = root / frozen["freeze"]["cohort"]["oracle_records"]
    oracle_rows = {
        row["state_hash"]: row
        for row in (
            json.loads(line) for line in oracle_path.read_text().splitlines() if line
        )
    }
    paired = evidence.get("pairs")
    if not isinstance(paired, dict):
        raise ValueError("evidence_arm_set_mismatch")
    normalized = validate_paired_evidence(paired, members, oracle_rows)
    artifact_root = (
        root / "docs/data/seed512-policy-strength-evaluation/runtime-artifacts"
    )
    configure_evaluation_runtime()
    evaluators = {
        name: arena.ArtifactEvaluator(artifact_root / name)
        for name in ("P-E4", "T512", "initializer")
    }

    def replay_search(
        arm: str, _key: str, member: dict[str, Any], seed: int
    ) -> dict[str, Any]:
        policy_name = "P-E4" if arm == "control" else "T512"
        composed = arena.ComposedArtifactEvaluator(
            evaluators[policy_name],
            evaluators["initializer"],
            policy_source="current",
            value_source="candidate",
        )
        result = arena.evaluate_artifact_position(
            evaluator=composed,
            state=member["state"],
            simulations=128,
            seed=seed,
            c_puct=1.25,
            search_options={
                "fpu_mode": "zero",
                "normalize_values": False,
                "reuse_subtree": False,
                "root_policy_mode": "deterministic",
                "root_temperature": 0.0,
                "tactical_root_bias": 0.0,
            },
            exact_root_solve_threshold=None,
            exact_solve_stone_threshold=None,
        )
        return {
            "selected_action": int(result["selected_move"]),
            "legal_moves": [int(move) for move in result["legal_moves"]],
            "visit_counts": [int(value) for value in result["visits"]],
            "child_stats": [
                {
                    "move": int(item["move"]),
                    "visits": int(item["visits"]),
                    "q_value": float(item["q_value"]),
                }
                for item in result["child_stats"]
            ],
            "network_priors": [float(value) for value in result["policy"]],
        }

    verify_search_replays(paired, members, replay_search)
    observed_gate = reconstruct_gate(normalized)
    if evidence.get("gate") != observed_gate:
        raise ValueError("gate_reconstruction_mismatch")
    return {"status": "verified", "roots": len(members), "gate": observed_gate}


def verify_gate_only(evidence: dict[str, Any]) -> dict[str, Any]:
    """Read-only synthetic fixture path used to test gate reconstruction."""
    if evidence.get("schema") != "seed512-policy-strength-evidence-v1":
        raise ValueError("evidence_schema_mismatch")
    observed_gate = reconstruct_gate(evidence["pairs"])
    if evidence.get("gate") != observed_gate:
        raise ValueError("gate_reconstruction_mismatch")
    return observed_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--trusted-authorization-sha256", required=True)
    args = parser.parse_args()
    result = verify_evidence(
        args.root,
        json.loads(args.evidence.read_bytes()),
        args.authorization,
        args.trusted_authorization_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
