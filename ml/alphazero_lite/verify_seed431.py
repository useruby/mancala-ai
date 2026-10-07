"""Replay archived seed429 arena games to verify every provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    search_configuration_ledger_record,
    seed_identity_ledger_record,
    stable_hash,
    verify_provenance_ledgers,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed430 import (
    DATA_REL,
    _require,
    rows,
    sha,
    verify as verify_430,
)


def canonical_game_state_hash(game: KalahGame) -> str:
    """Match the frozen arena's hash of the complete game state."""
    return stable_hash(game.to_state())


def relative_action_for_absolute(
    game: KalahGame, absolute: int, *, lane: str, game_index: int, ply: int
) -> int:
    actor = int(game.current_player)
    _require(0 <= absolute < 12, f"{lane}_absolute_pit_out_of_range:{game_index}:{ply}")
    _require(absolute // 6 == actor, f"{lane}_trajectory_wrong_side:{game_index}:{ply}")
    relative = absolute - 6 * actor
    _require(
        relative in game.possible_moves(),
        f"{lane}_trajectory_empty_or_illegal:{game_index}:{ply}",
    )
    return relative


def replay_lane(
    root: Path,
    registration: dict[str, Any],
    runtime_binding: dict[str, Any],
    lane: str,
    games: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> int:
    """Validate ordered ledgers against each action in the archived games."""
    data = root / DATA_REL
    suite = rows(root / registration["suite_path"])
    evaluation = registration["evaluation"]
    candidate_weights = runtime_binding["candidates"][lane]["artifact_files"][
        "weights.json"
    ]
    opponent_weights = runtime_binding["opponent_files"]["weights.json"]
    profile_hash = json.loads(
        (data / "arena-evidence" / lane / f"{lane}-arena.json").read_text()
    )["notes"]["search_profile"]["hash"]
    budget_pair = (
        f"{evaluation['simulations_per_side']}:{evaluation['simulations_per_side']}"
    )
    records_seen = 0

    _require(
        len(seeds) == len(configurations) == len(outcomes),
        f"{lane}_ledger_cardinality_disagreement",
    )
    # First validate self-hashes; the replay below independently validates their context.
    verify_provenance_ledgers(
        seed_identity_ledger=seeds,
        search_configuration_ledger=configurations,
        search_outcome_ledger=outcomes,
    )
    seed_keys: set[tuple[int, int]] = set()
    config_contexts: set[str] = set()
    outcome_keys: set[tuple[int, int]] = set()

    for game_entry in games:
        game_index = int(game_entry["game_index"])
        opening_index = int(game_entry["opening_index"])
        seat = int(game_entry["challenger_player"])
        _require(
            game_index == opening_index * 2 + int(game_entry["game_within_opening"]),
            f"{lane}_game_identity_invalid:{game_index}",
        )
        opening = suite[opening_index]
        game = KalahGame.from_state(suites.INITIAL_STATE)
        for prefix_action in opening["prefix_moves"]:
            legal_prefix = game.possible_moves()
            _require(
                prefix_action in legal_prefix,
                f"{lane}_opening_action_illegal:{game_index}",
            )
            _require(
                game.move(game.pit_index(int(prefix_action))),
                f"{lane}_opening_action_rejected:{game_index}",
            )
        _require(
            game.to_state() == opening["state"],
            f"{lane}_opening_state_mismatch:{game_index}",
        )
        _require(
            canonical_game_state_hash(game) == opening["state_hash"],
            f"{lane}_opening_hash_mismatch:{game_index}",
        )
        _require(
            game_entry["opening_state_hash"] == opening["state_hash"],
            f"{lane}_game_opening_hash_mismatch:{game_index}",
        )
        _require(
            game_entry["opening_prefix_moves"] == opening["prefix_moves"],
            f"{lane}_game_opening_prefix_mismatch:{game_index}",
        )

        trajectory = [
            int(value) for value in game_entry["trajectory"].split(",") if value
        ]
        _require(
            len(trajectory) == int(game_entry["game_length"]),
            f"{lane}_trajectory_length_invalid:{game_index}",
        )
        for ply, absolute in enumerate(trajectory):
            _require(
                not game.over(),
                f"{lane}_trajectory_continues_after_terminal:{game_index}:{ply}",
            )
            actor = int(game.current_player)
            relative = relative_action_for_absolute(
                game, absolute, lane=lane, game_index=game_index, ply=ply
            )
            role = "challenger" if actor == seat else "current"
            identity = seed_identity_ledger_record(
                contract_version=SEED_CONTRACT_VERSION,
                base_seed=evaluation["seed"],
                suite_sha256=registration["suite_sha256"],
                opening_index=opening_index,
                opening_state_hash=opening["state_hash"],
                challenger_player=seat,
                game_within_opening=int(game_entry["game_within_opening"]),
                ply=ply,
                canonical_current_state_hash=canonical_game_state_hash(game),
                acting_role=role,
                rng_stream_name="puct_search",
            )
            _require(
                records_seen < len(seeds),
                f"{lane}_missing_seed_record:{game_index}:{ply}",
            )
            seed, config, outcome = (
                seeds[records_seen],
                configurations[records_seen],
                outcomes[records_seen],
            )
            key = (game_index, ply)
            _require(
                key not in seed_keys, f"{lane}_duplicate_seed_record:{game_index}:{ply}"
            )
            _require(
                key not in outcome_keys,
                f"{lane}_duplicate_outcome_record:{game_index}:{ply}",
            )
            _require(
                seed == identity,
                f"{lane}_replayed_seed_identity_mismatch:{game_index}:{ply}",
            )
            _require(
                config["seed_context_hash"] == identity["seed_context_hash"],
                f"{lane}_configuration_context_mismatch:{game_index}:{ply}",
            )
            _require(
                config["seed_context_hash"] not in config_contexts,
                f"{lane}_duplicate_configuration_record:{game_index}:{ply}",
            )
            expected_config = search_configuration_ledger_record(
                seed_context_hash=identity["seed_context_hash"],
                simulations=evaluation["simulations_per_side"],
                effective_c_puct=evaluation["c_puct"],
                tactical_root_bias=evaluation["search_options"]["tactical_root_bias"],
                runtime_profile_hash=profile_hash,
                budget_pair=budget_pair,
                artifact_hash=candidate_weights
                if role == "challenger"
                else opponent_weights,
            )
            _require(
                config == expected_config,
                f"{lane}_configuration_registration_mismatch:{game_index}:{ply}",
            )
            _require(
                outcome["seed_context_hash"] == identity["seed_context_hash"]
                and int(outcome["game_index"]) == game_index
                and int(outcome["ply"]) == ply
                and outcome["acting_role"] == role,
                f"{lane}_outcome_context_mismatch:{game_index}:{ply}",
            )
            _require(
                int(outcome["selected_move"]) == relative,
                f"{lane}_selected_action_trajectory_mismatch:{game_index}:{ply}",
            )
            seed_keys.add(key)
            config_contexts.add(config["seed_context_hash"])
            outcome_keys.add(key)
            records_seen += 1
            # Trajectory values are already absolute pits; do not call pit_index here.
            _require(
                game.move(absolute),
                f"{lane}_trajectory_move_rejected:{game_index}:{ply}",
            )
        _require(game.over(), f"{lane}_trajectory_did_not_end:{game_index}")

    _require(
        records_seen == len(seeds),
        f"{lane}_extra_per_ply_records:{len(seeds) - records_seen}",
    )
    _require(
        len(seed_keys) == records_seen
        and len(config_contexts) == records_seen
        and len(outcome_keys) == records_seen,
        f"{lane}_duplicate_per_ply_records",
    )
    return records_seen


def verify_reconciliation(root: Path) -> dict[str, Any]:
    """Recompute every published game-entry digest from the receipt's chunk slices."""
    data = root / DATA_REL
    receipt = json.loads((data / "arena-resume-reconciliation.json").read_text())
    outcome = json.loads((data / "outcome-binding.json").read_text())
    lane_entries = {
        lane: rows(root / outcome["lanes"][lane]["games_path"]) for lane in ("A", "B")
    }
    _require(len(receipt["chunks"]) == 47, "reconciliation_chunk_count_invalid")
    checked = 0
    for chunk in receipt["chunks"]:
        lane = chunk["lane"]
        count = int(chunk["game_count"])
        verify_chunk_game_entries(chunk, lane_entries[lane])
        checked += count
    _require(checked == 1504, "reconciliation_pre_repair_coverage_invalid")
    return {
        "chunks": len(receipt["chunks"]),
        "pre_repair_games": checked,
        "result": "published game entries match the recorded digests",
        "original_chunk_report_bytes": "unavailable; not independently verified",
        "repair_timing": "not independently verified",
    }


def verify_chunk_game_entries(
    chunk: dict[str, Any], lane_entries: list[dict[str, Any]]
) -> None:
    """Check a recorded digest against its exact published lane-game slice."""
    lane = chunk["lane"]
    start, count = int(chunk["start_index"]), int(chunk["game_count"])
    selected = [
        entry
        for entry in lane_entries
        if start <= int(entry["game_index"]) < start + count
    ]
    _require(
        len(selected) == count, f"reconciliation_game_selection_invalid:{lane}:{start}"
    )
    _require(
        [int(entry["game_index"]) for entry in selected]
        == list(range(start, start + count)),
        f"reconciliation_game_coverage_invalid:{lane}:{start}",
    )
    digest = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _require(
        digest == chunk["game_entries_sha256"],
        f"reconciliation_game_entries_digest_mismatch:{lane}:{start}",
    )


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    base_result = verify_430(root)
    data = root / DATA_REL
    receipt_path = data / "seed431-verification-correction.json"
    receipt = json.loads(receipt_path.read_text())
    for field, path in (
        ("registration_sha256", data / "registration.json"),
        ("runtime_binding_sha256", data / "runtime-binding.json"),
        ("arena_correction_receipt_sha256", data / "arena-correction-receipt.json"),
        ("outcome_binding_sha256", data / "outcome-binding.json"),
        ("analysis_sha256", data / "analysis.json"),
    ):
        _require(receipt[field] == sha(path), f"seed431_binding_mismatch:{field}")
    _require(
        receipt["seed430_correction_receipt_sha256"]
        == sha(data / "seed430-verification-correction.json"),
        "seed431_binding_mismatch:seed430_correction_receipt_sha256",
    )
    for relative, digest in receipt["supplemental_sources"].items():
        _require(
            sha(root / relative) == digest, f"seed431_source_hash_mismatch:{relative}"
        )
    for relative, digest in receipt["frozen_source_sha256"].items():
        _require(
            sha(root / relative) == digest,
            f"seed431_frozen_source_hash_mismatch:{relative}",
        )
    registration = json.loads((data / "registration.json").read_text())
    runtime_binding = json.loads((data / "runtime-binding.json").read_text())
    outcome = json.loads((data / "outcome-binding.json").read_text())
    lane_counts: dict[str, int] = {}
    for lane in ("A", "B"):
        record = outcome["lanes"][lane]
        games = rows(root / record["games_path"])
        seeds = rows(root / record["seed_identity_ledger_path"])
        configs = rows(root / record["search_configuration_ledger_path"])
        searches = rows(root / record["search_outcome_ledger_path"])
        lane_counts[lane] = replay_lane(
            root, registration, runtime_binding, lane, games, seeds, configs, searches
        )
    reconciliation = verify_reconciliation(root)
    analysis = json.loads((data / "analysis.json").read_text())
    return {
        **base_result,
        "seed431_semantic_verification": "valid",
        "replayed_per_ply_records": lane_counts,
        "reconciliation": reconciliation,
        "original_results": {
            "games": 2048,
            "games_per_lane": 1024,
            "matched_openings": int(analysis["opening_clusters"]),
            "paired_B_minus_A_mean": analysis["primary_mean_B_minus_A"],
            "paired_B_minus_A_95_interval": analysis["primary_95_percentile_interval"],
            "B_score_mean": analysis["B_opponent_score_mean"],
            "B_score_95_interval": analysis["B_opponent_score_95_percentile_interval"],
            "decision": analysis["decision"],
        },
        "hash_only_telemetry": [
            "visit_hash",
            "cache_hit",
            "search_executed",
            "evaluation_profile_hash",
            "search and timing telemetry",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
