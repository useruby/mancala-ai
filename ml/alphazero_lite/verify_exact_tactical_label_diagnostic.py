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
from ml.alphazero_lite.verify_broader_opening_coverage_results import (
    verify as verify_585_archive,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/exact-tactical-label-diagnostic"
EXPECTED_COHORT_SHA256 = (
    "6fcabe67cc4b36e41c1b084f330e9ac56524ac5e8a684160353b925750f583ab"
)
EXPECTED_EVIDENCE_SHA256 = (
    "8c0636d41dae6fa14796ed95da38a7510d772cbe05c5b3329bd506e2d9c61dca"
)
EXPECTED_FROZEN_DIAGNOSTIC_SOURCE_SHA256 = (
    "91b0a5710d05eda5adaeb743cee387a2bcfd057893be8f89fe3b5aa430eda1d5"
)
EXPECTED_FROZEN_ADAPTER_SOURCE_SHA256 = (
    "bf241fa772efd122355b0706fff9035ee40f6cb9ac913fdd829444f359d62f15"
)
EXPECTED_TABLEBASE_RECHECK_SHA256 = (
    "39efdb46a31b1f8abd50c696f70964886271957cff9a152c98c86f12bb6695db"
)
EXPECTED_PUBLISHED_RECHECK_DIAGNOSTIC_SOURCE_SHA256 = (
    "7dd49ba964748805dc86484e0ac561f99141ea41ba1c17e3c21dea17837fffd9"
)


def _stable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_rebuilt_source_roots(
    cohort: dict[str, Any],
    rebuilt_roots: list[dict[str, Any]],
    deduplication: dict[str, int],
) -> None:
    """Bind every frozen action/state row to the replay-authenticated archive."""
    if cohort.get("roots") != rebuilt_roots:
        raise ValueError("frozen_cohort_archive_reconstruction_mismatch")
    if cohort.get("deduplication") != deduplication:
        raise ValueError("frozen_cohort_archive_deduplication_mismatch")


def verify() -> dict[str, Any]:
    cohort = json.loads((DATA / "cohort.json").read_text())
    evidence = json.loads((DATA / "evidence.json").read_text())
    recheck = json.loads((DATA / "tablebase-recheck.json").read_text())
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
    recheck_hash = recheck.pop("recheck_sha256")
    if (
        recheck_hash != EXPECTED_TABLEBASE_RECHECK_SHA256
        or hashlib.sha256(_stable(recheck)).hexdigest() != recheck_hash
        or recheck.get("cohort_sha256") != cohort_hash
        or recheck.get("original_evidence_sha256") != evidence_hash
    ):
        raise ValueError("tablebase_recheck_digest_or_binding_mismatch")

    archive_report = verify_585_archive(
        diagnostic.ARCHIVE, diagnostic.ARCHIVE_EVIDENCE_SHA256
    )
    rebuilt_roots, rebuilt_deduplication = diagnostic._build_frozen_roots()
    verify_rebuilt_source_roots(cohort, rebuilt_roots, rebuilt_deduplication)
    if (
        cohort["solver_source_sha256"].get("diagnostic")
        != EXPECTED_FROZEN_DIAGNOSTIC_SOURCE_SHA256
    ):
        raise ValueError("frozen_diagnostic_source_identity_mismatch")

    if (
        _sha(ROOT / "ml/alphazero_lite/exact_kalah_solver.py")
        != cohort["solver_source_sha256"]["exact_kalah_solver"]
        or cohort["solver_source_sha256"]["native_root_tablebase_adapter"]
        != EXPECTED_FROZEN_ADAPTER_SOURCE_SHA256
    ):
        raise ValueError("frozen_solver_source_hash_mismatch")
    if (
        _sha(ROOT / "model-artifact/runtime/kalah_v1_21.kvtb")
        != cohort["tablebase_sha256"]
        or _sha(ROOT / "model-artifact/runtime/kalah_v1_tablebase")
        != cohort["tablebase_binary_sha256"]
    ):
        raise ValueError("authenticated_tablebase_hash_mismatch")
    current_recheck_sources = {
        "diagnostic": EXPECTED_PUBLISHED_RECHECK_DIAGNOSTIC_SOURCE_SHA256,
        "exact_kalah_solver": _sha(ROOT / "ml/alphazero_lite/exact_kalah_solver.py"),
        "native_root_tablebase_adapter": _sha(
            ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py"
        ),
    }
    if recheck["recheck_source_sha256"] != current_recheck_sources:
        raise ValueError("tablebase_recheck_solver_source_mismatch")

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

    expected_rechecks = [
        {
            "root_index": root["index"],
            "row_id": root["row_id"],
            "move": int(move),
            "old_margin": action["lower"],
        }
        for root, result in zip(roots, results, strict=True)
        for move, action in result["actions"].items()
        if action["exact"]
    ]
    checked_claims = recheck["claims"]
    if len(checked_claims) != len(expected_rechecks):
        raise ValueError("tablebase_recheck_claim_count_mismatch")
    for claim, expected in zip(checked_claims, expected_rechecks, strict=True):
        if any(claim.get(key) != value for key, value in expected.items()):
            raise ValueError("tablebase_recheck_source_claim_mismatch")
        if (
            claim["status"] != "exact"
            or claim["rechecked_margin"] != claim["old_margin"]
            or not claim["matches"]
            or claim["tablebase_calls"] < 1
            or claim["elapsed_seconds"] > 5.1
        ):
            raise ValueError("tablebase_recheck_not_certified")

    summary = diagnostic.verify()
    if (
        summary["certified_best_action_roots"] != counts["certified"]
        or summary["proven_inferior_archived_moves"] != counts["inferior"]
        or summary["outcome_downgrades"] != downgrades
    ):
        raise ValueError("summary_reconstruction_mismatch")
    return {
        "verified": True,
        "source_archive_pairs_replayed": archive_report[
            "pairs_replayed_and_reconciled"
        ],
        "current_deadline_tablebase_implementation_sha256": _sha(
            ROOT / "ml/alphazero_lite/exact_tactical_label_diagnostic.py"
        ),
        "tablebase_recheck_sha256": recheck_hash,
        "tablebase_rechecked_exact_claims": len(checked_claims),
        "cohort_sha256": cohort_hash,
        "evidence_sha256": evidence_hash,
        "counts": counts,
        "outcome_downgrades": downgrades,
        "planning_gate": summary["planning_gate"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
