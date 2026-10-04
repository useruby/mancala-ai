"""Post-execution replay audit for omitted seed398 outcome-ledger states.

This correction is explicitly retrospective; it does not repair the original
seed414 preregistration or imply that its exclusion proof was complete.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed414-root-budget-confirmation"
BASE = ROOT / "docs/data/seed398-policy-value-composition/opening-exclusion-proof.json"
LEDGER = (
    ROOT / "docs/data/seed398-policy-value-composition/validated-outcome-ledger.jsonl"
)
LEDGER_BINDING = (
    ROOT / "docs/data/seed398-policy-value-composition/evaluation-binding.json"
)
LEDGER_ANALYSIS = ROOT / "docs/data/seed398-policy-value-composition/analysis.json"
SUITE = DATA / "suite.jsonl"
ORIGINAL_REGISTRATION = DATA / "registration.json"
ORIGINAL_PROOF = DATA / "opening-exclusion-proof.json"
CORRECTED = DATA / "post-execution-corrected-exclusion-proof.json"
RECEIPT = DATA / "post-execution-audit-receipt.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_ledger_binding() -> None:
    analysis = json.loads(LEDGER_ANALYSIS.read_text())
    ledger_hash = digest(LEDGER)
    binding = json.loads(LEDGER_BINDING.read_text())
    if analysis.get("outcome_ledger_sha256") != ledger_hash:
        raise ValueError("validated_ledger_binding_mismatch")
    if binding.get("schema") != "seed398-policy-value-composition-binding-v1":
        raise ValueError("validated_ledger_publication_binding_schema_mismatch")
    if len(read_jsonl(LEDGER)) != 4096:
        raise ValueError("validated_ledger_game_count_mismatch")
    if binding.get("status") != "completed_4096_games":
        raise ValueError("validated_ledger_publication_status_mismatch")


def replay_ledger(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, int]]:
    if len(rows) != 4096:
        raise ValueError("ledger_row_count_mismatch")
    identities: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("opening_contract") != "arena_player_relative_v2":
            raise ValueError(f"opening_contract_mismatch:{index}")
        if not isinstance(row.get("trajectory"), str):
            raise ValueError(f"malformed_trajectory:{index}")
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = row.get("opening_prefix_moves")
        if not isinstance(prefix, list) or int(
            row.get("opening_applied_prefix_length", -1)
        ) != len(prefix):
            raise ValueError(f"malformed_opening_prefix:{index}")
        if apply_opening_moves(game, [int(move) for move in prefix]) != len(prefix):
            raise ValueError(f"illegal_opening_prefix:{index}")
        if suites.canonical_key(game.to_state()) != row.get("opening_state_hash"):
            raise ValueError(f"opening_state_hash_mismatch:{index}")
        identities.add(suites.canonical_key(game.to_state()))
        moves = row["trajectory"].split(",") if row["trajectory"] else []
        for ply, encoded in enumerate(moves):
            try:
                absolute = int(encoded)
            except ValueError as exc:
                raise ValueError(f"malformed_absolute_action:{index}:{ply}") from exc
            if (
                game.over()
                or not 0 <= absolute < 12
                or game.pit_owner(absolute) != game.current_player
            ):
                raise ValueError(f"trajectory_actor_or_terminal_mismatch:{index}:{ply}")
            if not game.move(absolute):
                raise ValueError(f"illegal_absolute_action:{index}:{ply}")
            identities.add(suites.canonical_key(game.to_state()))
        if not game.over() or int(row.get("game_length", -1)) != len(moves):
            raise ValueError(f"trajectory_not_terminal_or_length_mismatch:{index}")
        challenger = int(row["challenger_player"])
        margin = game.captured_seeds[challenger] - game.captured_seeds[1 - challenger]
        winner = "draw" if margin == 0 else "challenger" if margin > 0 else "current"
        if margin != int(row["margin"]) or winner != row["winner"]:
            raise ValueError(f"terminal_outcome_mismatch:{index}")
    return identities, {
        "games": len(rows),
        "additional_unique_identities": len(identities),
    }


def find_suite_collisions(
    identities: set[str], suite: list[dict[str, Any]]
) -> list[str]:
    suite_ids = {suites.canonical_key(row["state"]) for row in suite}
    return sorted(identities & suite_ids)


def build_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_ledger_binding()
    rows = read_jsonl(LEDGER)
    ledger_ids, counts = replay_ledger(rows)
    old = json.loads(ORIGINAL_PROOF.read_text())
    frozen = (
        json.loads(SUITE.read_text()) if SUITE.suffix == ".json" else read_jsonl(SUITE)
    )
    old_union = set(old["excluded_state_identities"])
    collisions = find_suite_collisions(ledger_ids, frozen)
    union = old_union | ledger_ids
    if len(union) != 171434 or len(union - old_union) != 63471 or collisions:
        raise ValueError("independent_audit_expected_counts_mismatch")
    proof = {
        "schema": "seed414-post-execution-corrected-exclusion-proof-v1",
        "audit_timing": "after_execution",
        "original_proof_sha256": digest(ORIGINAL_PROOF),
        "original_exclusion_union_count": len(old_union),
        "ledger_identity_count": len(ledger_ids),
        "additional_identity_count": len(union - old_union),
        "corrected_union_count": len(union),
        "corrected_excluded_identity_sha256": hashlib.sha256(
            "\n".join(sorted(union)).encode()
        ).hexdigest(),
        "excluded_state_identities": sorted(union),
        "ledger_state_identities": sorted(ledger_ids),
        "frozen_suite_collision_count": len(collisions),
        "frozen_suite_collisions": collisions,
    }
    receipt = {
        "schema": "seed414-post-execution-audit-receipt-v1",
        "audit_timing": "after_execution",
        "preregistration_exclusion_proof_was_complete": False,
        "decision": "confirmation_criteria_not_met",
        "bindings": {
            "registration_sha256": digest(ORIGINAL_REGISTRATION),
            "original_proof_sha256": digest(ORIGINAL_PROOF),
            "historical_proof_sha256": digest(BASE),
            "validated_outcome_ledger_sha256": digest(LEDGER),
            "validated_outcome_ledger_binding_sha256": digest(LEDGER_BINDING),
            "validated_outcome_ledger_published_hash_source_sha256": digest(
                LEDGER_ANALYSIS
            ),
            "suite_sha256": digest(SUITE),
            "corrected_proof_sha256": hashlib.sha256(
                (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "audit_source_sha256": digest(Path(__file__)),
        },
        "counts": counts
        | {
            "additional_identities": proof["additional_identity_count"],
            "corrected_union": len(union),
        },
        "collision_report": {
            "suite_state_count": len(frozen),
            "collision_count": len(collisions),
            "collisions": collisions,
        },
        "statement": "This audit occurred after execution. It corrects historical coverage only and does not claim the original preregistration exclusion proof was complete.",
    }
    return proof, receipt


def publish() -> None:
    proof, receipt = build_audit()
    for path, value in ((CORRECTED, proof), (RECEIPT, receipt)):
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        path.write_text(payload)


if __name__ == "__main__":
    publish()
