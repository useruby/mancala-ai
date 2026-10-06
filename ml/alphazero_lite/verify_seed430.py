"""Append-only semantic verification correction for the seed429 arena evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.verify_seed429_publication import verify as verify_429


DATA_REL = "docs/data/seed429-canonical-policy-normalization"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _opening_state(opening: dict[str, Any]) -> KalahGame:
    game = KalahGame.from_state(suites.INITIAL_STATE)
    for move in opening["prefix_moves"]:
        _require(
            not game.over() and move in game.possible_moves(), "opening_prefix_invalid"
        )
        _require(game.move(game.pit_index(int(move))), "opening_prefix_invalid")
    _require(
        suites.canonical_key(game.to_state()) == suites.canonical_key(opening["state"]),
        "opening_identity_invalid",
    )
    return game


def verify_portable_reconciliation(
    root: Path, reg: dict[str, Any], binding: dict[str, Any], correction: dict[str, Any]
) -> dict[str, Any]:
    data = root / DATA_REL
    receipt = json.loads((data / "arena-resume-reconciliation.json").read_text())
    for field, path in (
        ("registration_sha256", data / "registration.json"),
        ("runtime_binding_sha256", data / "runtime-binding.json"),
        ("correction_receipt_sha256", data / "arena-correction-receipt.json"),
    ):
        _require(receipt[field] == sha(path), f"reconciliation_{field}_mismatch")
    _require(
        receipt["training_results_sha256"] == sha(data / "training-results.json"),
        "reconciliation_training_mismatch",
    )
    _require(
        receipt["outcomes_unchanged"] is True and receipt["protocol_changed"] is False,
        "reconciliation_semantics_invalid",
    )
    _require(len(receipt["chunks"]) == 47, "reconciliation_chunk_count_invalid")
    counts = {
        lane: sum(int(c["game_count"]) for c in receipt["chunks"] if c["lane"] == lane)
        for lane in ("A", "B")
    }
    _require(
        counts == {"A": 1024, "B": 480} == receipt["lane_game_counts"],
        "reconciliation_lane_counts_invalid",
    )
    _require(
        receipt["completed_game_count_before_repair"] == 1504,
        "reconciliation_total_invalid",
    )
    for lane in ("A", "B"):
        chunks = [c for c in receipt["chunks"] if c["lane"] == lane]
        expected = list(range(0, counts[lane], 32))
        _require(
            [int(c["start_index"]) for c in chunks] == expected,
            f"reconciliation_{lane}_coverage_invalid",
        )
        _require(
            all(c["game_count"] == 32 for c in chunks),
            f"reconciliation_{lane}_chunk_size_invalid",
        )
    return {
        "chunks": 47,
        "pre_repair_games": 1504,
        "byte_level_chunk_claims": "unverifiable: run-local chunk and report bytes are not published",
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    base_result = verify_429(root, require_complete=True)
    data = root / DATA_REL
    reg_path, runtime_path = data / "registration.json", data / "runtime-binding.json"
    correction_path = data / "arena-correction-receipt.json"
    reg, runtime, correction = (
        json.loads(p.read_text()) for p in (reg_path, runtime_path, correction_path)
    )
    supplemental_receipt_path = data / "seed430-verification-correction.json"
    supplemental_receipt = json.loads(supplemental_receipt_path.read_text())
    _require(
        supplemental_receipt["registration_sha256"] == sha(reg_path),
        "seed430_original_registration_mismatch",
    )
    _require(
        supplemental_receipt["runtime_binding_sha256"] == sha(runtime_path),
        "seed430_original_runtime_mismatch",
    )
    _require(
        supplemental_receipt["arena_correction_receipt_sha256"] == sha(correction_path),
        "seed430_arena_correction_mismatch",
    )
    for rel, digest in supplemental_receipt["supplemental_sources"].items():
        _require(sha(root / rel) == digest, f"seed430_source_hash_mismatch:{rel}")
    outcome_path = data / "outcome-binding.json"
    outcome = json.loads(outcome_path.read_text())
    reg_hash, runtime_hash, correction_hash = (
        sha(reg_path),
        sha(runtime_path),
        sha(correction_path),
    )
    _require(outcome.get("status") == "completed_2048_games", "outcome_status_invalid")
    _require(set(outcome["lanes"]) == {"A", "B"}, "outcome_lane_set_invalid")
    _require(
        outcome["registration_sha256"] == reg_hash, "outcome_registration_mismatch"
    )
    _require(
        outcome["runtime_binding_sha256"] == runtime_hash,
        "outcome_runtime_binding_mismatch",
    )
    _require(
        outcome["correction_receipt_sha256"] == correction_hash,
        "outcome_correction_mismatch",
    )
    analysis = json.loads((data / "analysis.json").read_text())
    _require(
        analysis["registration_sha256"] == reg_hash, "analysis_registration_mismatch"
    )
    _require(
        analysis["outcome_binding_sha256"] == sha(outcome_path),
        "analysis_outcome_binding_hash_mismatch",
    )
    suite = rows(root / reg["suite_path"])
    ledger = rows(data / "outcome-ledger.jsonl")
    _require(len(ledger) == 2048, "ledger_count_invalid")
    runtime_contract = runtime["runtime_contract"]
    game_by_lane: dict[str, dict[int, dict[str, Any]]] = {}
    for lane in ("A", "B"):
        record = outcome["lanes"][lane]
        paths = {
            "report": record["report_path"],
            "games": record["games_path"],
            "seed_identities": record["seed_identity_ledger_path"],
            "search_configurations": record["search_configuration_ledger_path"],
            "search_outcomes": record["search_outcome_ledger_path"],
        }
        hashes = {
            "report": "report_sha256",
            "games": "games_sha256",
            "seed_identities": "seed_identity_ledger_sha256",
            "search_configurations": "search_configuration_ledger_sha256",
            "search_outcomes": "search_outcome_ledger_sha256",
        }
        for name, rel in paths.items():
            _require(
                sha(root / rel) == record[hashes[name]], f"{lane}_{name}_hash_mismatch"
            )
        report = json.loads((root / paths["report"]).read_text())
        games, seeds = (
            rows(root / paths["games"]),
            rows(root / paths["seed_identities"]),
        )
        configs, searches = (
            rows(root / paths["search_configurations"]),
            rows(root / paths["search_outcomes"]),
        )
        n = 1024
        _require(
            record["game_count"] == n and len(games) == n,
            f"{lane}_game_coverage_invalid",
        )
        _require(
            len(seeds) == len(configs) == len(searches),
            f"{lane}_search_record_count_invalid",
        )
        _require(
            len(
                {
                    (
                        r["opening_index"],
                        r["challenger_player"],
                        r["game_within_opening"],
                        r["ply"],
                        r["acting_role"],
                    )
                    for r in seeds
                }
            )
            == len(seeds),
            f"{lane}_seed_duplicates",
        )
        _require(
            len({r["seed_context_hash"] for r in configs}) == len(configs),
            f"{lane}_config_duplicates",
        )
        _require(
            len(
                {(r["seed_context_hash"], r["ply"], r["acting_role"]) for r in searches}
            )
            == len(searches),
            f"{lane}_search_duplicates",
        )
        seed_contexts = {r["seed_context_hash"] for r in seeds}
        _require(
            len(record["seed_contexts"]) == 1024
            and len(set(record["seed_contexts"])) == 1024
            and set(record["seed_contexts"]) <= seed_contexts,
            f"{lane}_seed_context_coverage_invalid",
        )
        _require(
            {r["seed_context_hash"] for r in configs} <= seed_contexts,
            f"{lane}_configuration_seed_link_invalid",
        )
        _require(
            {r["seed_context_hash"] for r in searches} <= seed_contexts,
            f"{lane}_search_seed_link_invalid",
        )
        for seed in seeds:
            _require(
                seed["base_seed"] == reg["evaluation"]["seed"]
                and seed["contract_version"] == reg["evaluation"]["seed_contract"]
                and seed["suite_sha256"] == reg["suite_sha256"],
                f"{lane}_seed_contract_invalid",
            )
        for config in configs:
            _require(
                config["simulations"] == reg["evaluation"]["simulations_per_side"]
                and config["effective_c_puct"] == reg["evaluation"]["c_puct"],
                f"{lane}_search_configuration_invalid",
            )
        by_game = {}
        for g in games:
            idx, opening_id, seat = (
                int(g["game_index"]),
                int(g["opening_index"]),
                int(g["challenger_player"]),
            )
            _require(
                idx not in by_game and idx == len(by_game),
                f"{lane}_game_indices_invalid",
            )
            _require(
                opening_id == idx // 2 and seat == idx % 2,
                f"{lane}_seat_pairing_invalid",
            )
            opening = suite[opening_id]
            _require(
                g["opening_prefix_moves"] == opening["prefix_moves"],
                f"{lane}_opening_prefix_mismatch",
            )
            _opening_state(opening)
            _require(
                g["opening_state_hash"] == opening["state_hash"],
                f"{lane}_opening_state_hash_mismatch",
            )
            by_game[idx] = g
        _require(report["games_played"] == n, f"{lane}_report_game_count_invalid")
        outcomes = {"wins": 0, "draws": 0, "losses": 0}
        scores = []
        for g in games:
            result = g["winner"]
            outcomes[
                {"challenger": "wins", "current": "losses", "draw": "draws"}[result]
            ] += 1
            scores.append(
                1.0 if result == "challenger" else 0.5 if result == "draw" else 0.0
            )
        _require(
            (report["wins"], report["draws"], report["losses"])
            == (outcomes["wins"], outcomes["draws"], outcomes["losses"]),
            f"{lane}_report_outcomes_invalid",
        )
        _require(report["score"] == sum(scores) / n, f"{lane}_report_score_invalid")
        notes = report["notes"]
        profile = notes["search_profile"]
        _require(
            notes["base_seed"] == reg["evaluation"]["seed"]
            and notes["seed_contract"] == reg["evaluation"]["seed_contract"]
            and notes["suite_sha256"] == reg["suite_sha256"],
            f"{lane}_report_seed_contract_invalid",
        )
        _require(
            profile["simulations"] == reg["evaluation"]["simulations_per_side"]
            and profile["c_puct"] == reg["evaluation"]["c_puct"]
            and profile["search_options"] == reg["evaluation"]["search_options"],
            f"{lane}_report_search_protocol_invalid",
        )
        for key in (
            "runtime_search_policy_mode",
            "runtime_search_policy_sha256",
            "exact_root_solve_threshold",
            "exact_root_native_probe",
            "exact_root_native_probe_sha256",
            "exact_root_tablebase",
            "exact_root_tablebase_sha256",
        ):
            _require(
                profile[key] == runtime_contract[key],
                f"{lane}_runtime_setting_mismatch:{key}",
            )
        lane_ledger = [r for r in ledger if r["lane"] == lane]
        _require(len(lane_ledger) == n, f"{lane}_final_ledger_count_invalid")
        for row in lane_ledger:
            idx = int(row["game"]["game_index"])
            _require(
                idx in by_game and row["game"] == by_game[idx],
                f"{lane}_archived_game_ledger_disagreement",
            )
            _require(
                int(row["opening_id"]) == idx // 2
                and row["opponent_score"] == scores[idx],
                f"{lane}_ledger_score_or_opening_invalid",
            )
        game_by_lane[lane] = by_game
    reconciliation = verify_portable_reconciliation(root, reg, runtime, correction)
    return {
        **base_result,
        "supplemental_verification": "valid",
        "reconciliation": reconciliation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
