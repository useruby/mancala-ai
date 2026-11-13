"""Independent, read-only verifier for the frozen tactical-label diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import exact_tactical_label_diagnostic as diagnostic
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/exact-tactical-label-diagnostic"
EXPECTED_COHORT_SHA256 = (
    "6fcabe67cc4b36e41c1b084f330e9ac56524ac5e8a684160353b925750f583ab"
)
EXPECTED_EVIDENCE_SHA256 = (
    "8c0636d41dae6fa14796ed95da38a7510d772cbe05c5b3329bd506e2d9c61dca"
)


def _stable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    cohort = json.loads((DATA / "cohort.json").read_text())
    evidence = json.loads((DATA / "evidence.json").read_text())
    cohort_hash = cohort.pop("cohort_sha256")
    evidence_hash = evidence.pop("evidence_sha256")
    if (
        cohort_hash != EXPECTED_COHORT_SHA256
        or hashlib.sha256(_stable(cohort)).hexdigest() != cohort_hash
    ):
        raise ValueError("external_or_recomputed_cohort_digest_mismatch")
    if (
        evidence_hash != EXPECTED_EVIDENCE_SHA256
        or hashlib.sha256(_stable(evidence)).hexdigest() != evidence_hash
    ):
        raise ValueError("external_or_recomputed_evidence_digest_mismatch")
    if evidence.get("cohort_sha256") != cohort_hash:
        raise ValueError("evidence_cohort_binding_mismatch")

    source_paths = {
        "diagnostic": ROOT / "ml/alphazero_lite/exact_tactical_label_diagnostic.py",
        "exact_kalah_solver": ROOT / "ml/alphazero_lite/exact_kalah_solver.py",
        "native_root_tablebase_adapter": ROOT
        / "ml/alphazero_lite/native_exact_root_tablebase.py",
    }
    if {name: _sha(path) for name, path in source_paths.items()} != cohort[
        "solver_source_sha256"
    ]:
        raise ValueError("solver_source_hash_mismatch")
    if (
        _sha(ROOT / "model-artifact/runtime/kalah_v1_21.kvtb")
        != cohort["tablebase_sha256"]
        or _sha(ROOT / "model-artifact/runtime/kalah_v1_tablebase")
        != cohort["tablebase_binary_sha256"]
    ):
        raise ValueError("authenticated_tablebase_hash_mismatch")

    roots = cohort["roots"]
    results = evidence["roots"]
    if len(roots) != 256 or len(results) != len(roots):
        raise ValueError("cohort_size_mismatch")
    canonical_ids: set[str] = set()
    float32_ids: set[str] = set()
    counts = {"certified": 0, "inferior": 0, "unresolved": 0}
    downgrades = {"win_to_draw": 0, "win_to_loss": 0, "draw_to_loss": 0}
    for root, result in zip(roots, results, strict=True):
        state = root["state"]
        if (
            root["active_stones"] <= 32
            or root["canonical_identity"] != canonical_identity(state)
            or root["production_float32_identity"] != production_float32_identity(state)
            or root["archived_move"] not in root["legal_moves"]
        ):
            raise ValueError("frozen_root_identity_or_eligibility_mismatch")
        if (
            root["canonical_identity"] in canonical_ids
            or root["production_float32_identity"] in float32_ids
        ):
            raise ValueError("duplicate_root_identity")
        canonical_ids.add(root["canonical_identity"])
        float32_ids.add(root["production_float32_identity"])
        if (
            result["index"] != root["index"]
            or result["archived_move"] != root["archived_move"]
        ):
            raise ValueError("per_root_binding_mismatch")
        if result["elapsed_seconds"] > root["budget_seconds"] + 0.1:
            raise ValueError("root_budget_exceeded")
        actions = result["actions"]
        if set(actions) != {str(move) for move in root["legal_moves"]}:
            raise ValueError("legal_action_evidence_mismatch")
        if any(
            action["exact"] != (action["lower"] == action["upper"])
            for action in actions.values()
            if action["lower"] is not None and action["upper"] is not None
        ):
            raise ValueError("exact_margin_interval_mismatch")
        exact = {
            int(move): action["lower"]
            for move, action in actions.items()
            if action["exact"]
        }
        sign = 1 if int(state["current_player"]) == 0 else -1
        archived = actions[str(root["archived_move"])]
        better = (
            [
                move
                for move, margin in exact.items()
                if move != root["archived_move"]
                and sign * margin > sign * archived["upper"]
            ]
            if archived["exact"]
            else []
        )
        if result["proving_moves"] != better or result["proven_inferior"] != bool(
            better
        ):
            raise ValueError("inferiority_claim_not_reconstructed")
        certified = len(exact) == len(actions)
        if result["certified_best_action"] != certified:
            raise ValueError("best_action_certification_mismatch")
        if certified:
            best = (max if sign == 1 else min)(exact.values())
            if result["best_margin"] != best:
                raise ValueError("best_margin_mismatch")
            counts["certified"] += 1
        else:
            counts["unresolved"] += 1
        if better:
            counts["inferior"] += 1
            actual = archived["lower"]
            actual_player = sign * actual
            best_player = sign * result["best_margin"]
            if best_player > 0 and actual_player == 0:
                downgrades["win_to_draw"] += 1
            elif best_player > 0 and actual_player < 0:
                downgrades["win_to_loss"] += 1
            elif best_player == 0 and actual_player < 0:
                downgrades["draw_to_loss"] += 1

    summary = diagnostic.verify()
    if (
        summary["certified_best_action_roots"] != counts["certified"]
        or summary["proven_inferior_archived_moves"] != counts["inferior"]
        or summary["outcome_downgrades"] != downgrades
    ):
        raise ValueError("summary_reconstruction_mismatch")
    return {
        "verified": True,
        "cohort_sha256": cohort_hash,
        "evidence_sha256": evidence_hash,
        "counts": counts,
        "outcome_downgrades": downgrades,
        "planning_gate": summary["planning_gate"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
