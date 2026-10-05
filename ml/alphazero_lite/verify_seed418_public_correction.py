"""Read-only verifier for published seed418 diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed418_cohort_replay import replay_cohort
from ml.alphazero_lite.seed418_analysis import analyze, choose_exact_action
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_bound_files(data: Path, binding: dict[str, str]) -> None:
    for name, expected_hash in binding.items():
        if digest(data / name) != expected_hash:
            raise ValueError(f"published_evidence_hash_mismatch:{name}")


def verify_receipt_identities(data: Path, root: Path, receipt: dict[str, Any]) -> None:
    if receipt.get("accepted_registration_sha256") != (
        "80765bdb69541b1a74e657f3130dcfac144916c484b2a9ff07f86ad637974ccb"
    ):
        raise ValueError("accepted_registration_identity_invalid")
    if receipt.get("original_execution_freeze_independently_verifiable") is not False:
        raise ValueError("historical_provenance_status_invalid")
    if receipt.get("accepted_registration_archive") is not None:
        raise ValueError("unavailable_registration_must_not_be_reconstructed")
    provenance = receipt.get("provenance_status", "").lower()
    if "unavailable" not in provenance or "independently" not in provenance:
        raise ValueError("historical_provenance_limitation_not_disclosed")
    expected_paths = {
        "post_execution_publication_registration": "preregistration.json",
        "completed_checkpoint": "execution-checkpoint-v2.jsonl",
        "manifest": "cohort-manifest.json",
        "analysis": "analysis.json",
        "existing_evidence_binding": "evidence-binding.json",
    }
    for key, name in expected_paths.items():
        entry = receipt[key]
        if entry.get("path") != name or digest(data / name) != entry.get("sha256"):
            raise ValueError(f"correction_receipt_identity_mismatch:{key}")
    for name, expected in receipt["published_records"].items():
        if digest(data / name) != expected:
            raise ValueError(f"correction_receipt_identity_mismatch:{name}")
    if receipt["completed_checkpoint"].get("rows") != 128:
        raise ValueError("correction_receipt_checkpoint_count_invalid")
    for relative, expected in receipt["new_verification_sources"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"correction_verifier_source_hash_mismatch:{relative}")


def verify(root: Path = ROOT) -> dict[str, Any]:
    data = root / "docs/data/seed418-native-root-handoff"
    prereg = json.loads((data / "preregistration.json").read_text())
    manifest_path = data / "cohort-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = root / "docs/data/seed416-policy-target-softening"
    suite_path, ledger_path = (
        source / "openings-v3.jsonl",
        source / "outcome-ledger.jsonl",
    )
    if (
        digest(suite_path) != prereg["source_suite_sha256"]
        or digest(ledger_path) != prereg["source_ledger_sha256"]
    ):
        raise ValueError("source_evidence_hash_mismatch")
    receipt_path = data / "correction-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("correction_receipt_missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "seed418-provenance-correction-v1":
        raise ValueError("correction_receipt_invalid")
    for relative, expected in prereg["sources"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    verify_receipt_identities(data, root, receipt)
    if digest(manifest_path) != prereg["manifest_sha256"]:
        raise ValueError("cohort_manifest_hash_mismatch")
    suite = read_jsonl(suite_path)
    source_rows = read_jsonl(ledger_path)
    expected = replay_cohort(suite, source_rows)
    if manifest["states"] != expected:
        raise ValueError("cohort_replay_mismatch")
    search = read_jsonl(data / "search-records.jsonl")
    oracle = read_jsonl(data / "oracle-records.jsonl")
    if len(search) != 128 or len(oracle) != 128:
        raise ValueError("case_count_invalid")
    checkpoint = read_jsonl(data / "execution-checkpoint-v2.jsonl")
    if len(checkpoint) != 128:
        raise ValueError("checkpoint_case_count_invalid")
    if manifest.get("schema") != "seed418-cohort-manifest-v1":
        raise ValueError("cohort_manifest_schema_invalid")
    if len(manifest.get("states", [])) != 128:
        raise ValueError("manifest_case_count_invalid")
    state_hashes = [row["state_hash"] for row in manifest["states"]]
    opening_indices = [row["opening_index"] for row in manifest["states"]]
    if len(set(state_hashes)) != 128 or len(set(opening_indices)) != 128:
        raise ValueError("manifest_duplicate_case")
    rows = []
    for item, srow, orow in zip(manifest["states"], search, oracle, strict=True):
        index = len(rows)
        checkpoint_row = checkpoint[index]
        if checkpoint_row.get("search") != srow or checkpoint_row.get("oracle") != orow:
            raise ValueError("checkpoint_published_pair_mismatch")
        if checkpoint_row.get("preregistration_sha256") != receipt.get(
            "accepted_registration_sha256"
        ):
            raise ValueError("checkpoint_registration_identity_invalid")
        if (
            item["state_hash"] != srow["state_hash"]
            or item["state_hash"] != orow["state_hash"]
        ):
            raise ValueError("case_state_binding_mismatch")
        legal = sorted(int(move) for move in orow["legal_moves"])
        margins = {
            int(move): int(value) for move, value in orow["action_margins"].items()
        }
        if legal != sorted(margins) or orow.get("coverage_complete") is not True:
            raise ValueError("exact_legal_action_coverage_invalid")
        best_margin = max(margins.values())
        expected_optimal_moves = [
            move for move in legal if margins[move] == best_margin
        ]
        if (
            sorted(int(move) for move in orow["optimal_moves"])
            != expected_optimal_moves
        ):
            raise ValueError("optimal_action_set_invalid")
        if sorted(int(move) for move in srow["legal_moves"]) != legal:
            raise ValueError("search_oracle_legal_actions_mismatch")
        game = KalahGame.from_state(item["state"])
        if (
            game.over()
            or sum(game.pits) != int(item["active_pit_stones"])
            or sorted(game.possible_moves()) != legal
            or int(orow["root_player"]) != game.current_player
            or int(srow["search_selected_move"]) not in legal
        ):
            raise ValueError("case_state_or_action_invalid")
        seed, seed_context_hash = derive_search_seed(**srow["seed_context"])
        expected_context = {
            "contract_version": "azlite_eval_seed_v2",
            "base_seed": 418,
            "suite_sha256": prereg["source_suite_sha256"],
            "opening_index": item["opening_index"],
            "opening_state_hash": item["source_opening_state_hash"],
            "challenger_player": 0,
            "game_within_opening": 0,
            "ply": item["decision_ply"],
            "canonical_current_state_hash": item["state_hash"],
            "acting_role": "challenger" if game.current_player == 0 else "current",
        }
        if (
            srow["seed_context"] != expected_context
            or int(srow["seed"]) != seed
            or srow["seed_context_hash"] != seed_context_hash
            or srow["seed_context"]["opening_state_hash"]
            != item["source_opening_state_hash"]
            or srow["seed_context"]["canonical_current_state_hash"]
            != item["state_hash"]
        ):
            raise ValueError("search_seed_context_invalid")
        if int(orow["selected_move"]) != choose_exact_action(
            margins, [float(value) for value in orow["network_priors"]], legal
        ):
            raise ValueError("exact_action_tie_rule_invalid")
        if len(orow["network_priors"]) != 6 or any(
            not math.isfinite(float(value)) for value in orow["network_priors"]
        ):
            raise ValueError("network_priors_invalid")
        for latency_name in (
            "native_decision_latency_ms",
            "native_probe_latency_ms",
            "neural_tie_break_latency_ms",
        ):
            if (
                not math.isfinite(float(orow[latency_name]))
                or float(orow[latency_name]) < 0
            ):
                raise ValueError("native_latency_invalid")
        if not math.isclose(
            float(orow["native_decision_latency_ms"]),
            float(orow["native_probe_latency_ms"])
            + float(orow["neural_tie_break_latency_ms"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("native_latency_components_mismatch")
        rows.append(
            {
                **item,
                **srow,
                "legal_moves": legal,
                "action_margins": orow["action_margins"],
                "native_decision_latency_ms": orow["native_decision_latency_ms"],
                "coverage_complete": True,
            }
        )
    analysis = analyze(rows)
    published = json.loads((data / "analysis.json").read_text())
    if analysis != published:
        raise ValueError("analysis_recomputation_mismatch")
    latency = json.loads((data / "latency-accounting.json").read_text())
    if (
        latency["warm_native_decision_mean_ms"]
        != analysis["native_decision_latency_ms"]["mean"]
        or latency["warm_native_decision_p95_ms"]
        != analysis["native_decision_latency_ms"]["p95_nearest_rank"]
        or latency.get("scope")
        != "warm adapter query plus neural tie-breaking; process startup/prewarm excluded and reported separately"
    ):
        raise ValueError("latency_accounting_mismatch")
    binding = json.loads((data / "evidence-binding.json").read_text())
    verify_bound_files(data, binding)
    return {
        "valid": True,
        "cases": 128,
        "decision": analysis["decision"],
        "mean_margin_regret": analysis["mean_final_margin_regret"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
