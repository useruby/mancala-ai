"""Synthetic controls for the seed511 freeze, routing, resume, and verifier."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from ml.alphazero_lite import seed511_fixed_composition_screen as protocol
from ml.alphazero_lite import verify_seed511_fixed_composition_screen as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


def _opening() -> dict:
    state = dict(protocol.INITIAL_STATE)
    return {
        "opening_index": 0,
        "opening_contract": "arena_player_relative_v2",
        "prefix_moves": [],
        "source_prefix_moves_absolute": [],
        "state": state,
        "state_hash": hashlib.sha256(protocol.canonical_json(state)).hexdigest(),
    }


def _pair(root: Path) -> tuple[dict, list[dict]]:
    out = root / protocol.OUT / "batches"
    out.mkdir(parents=True)
    opening = _opening()
    runtime_assets = {
        "exact_root_native_probe": {"artifact": "native-probe", "sha256": "probe-hash"},
        "exact_root_tablebase": {"artifact": "tablebase", "sha256": "tablebase-hash"},
    }
    (root / protocol.OUT / "registration.json").write_text(
        json.dumps(
            {
                "components": {},
                "runtime_assets": runtime_assets,
                "runtime": {"search_options": {"fpu_mode": "zero"}},
            }
        )
    )
    opening_path = out / "opening-0000.jsonl"
    opening_path.write_bytes(protocol.canonical_json(opening) + b"\n")
    game = KalahGame.from_state(protocol.INITIAL_STATE)
    rng = random.Random(4)
    trajectory = []
    while not game.over() and len(trajectory) < 200:
        move = game.pit_index(rng.choice(game.possible_moves()))
        trajectory.append(move)
        assert game.move(move)
    assert game.over()
    entries = []
    for seat in (0, 1):
        margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
        entries.append(
            {
                "challenger_player": seat,
                "trajectory": ",".join(map(str, trajectory)),
                "game_length": len(trajectory),
                "winner": "challenger"
                if margin > 0
                else "current"
                if margin < 0
                else "draw",
                "margin": margin,
            }
        )
    game_path = out / "games-0000.jsonl"
    game_path.write_text("".join(json.dumps(row) + "\n" for row in entries))
    report = {
        "games_played": 2,
        "notes": {
            "challenger_simulations": 128,
            "current_simulations": 128,
            "workers_used": 1,
            "workers_requested": 1,
            "base_seed": 511,
            "seed_contract": "azlite_eval_seed_v2",
            "search_profile": {
                "c_puct": 1.25,
                "exact_leaf_solve_mode": "disabled",
                "exact_root_objective": "final_score_margin",
                "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
                "search_options": {"fpu_mode": "zero"},
            },
            "exact_root_solve_threshold": 16,
            "exact_leaf_solve_mode": "disabled",
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_root_native_probe_sha256": "probe-hash",
            "exact_root_tablebase_sha256": "tablebase-hash",
            "search_options": {"fpu_mode": "zero"},
        },
        "seed511_composition": {
            "policy_source": "E4",
            "value_source": "initializer",
            "opponent": "complete initializer artifact",
            "components": {},
            "runtime_assets": runtime_assets,
        },
    }
    report_path = out / "report-0000.json"
    report_path.write_text(json.dumps(report))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    ledger = [
        {
            "opening_index": 0,
            "challenger_player": row["challenger_player"],
            "winner": row["winner"],
            "batch_game_position": pos,
            "batch_game_sha256": digest(game_path),
            "batch_report_sha256": digest(report_path),
            "batch_opening_sha256": digest(opening_path),
        }
        for pos, row in enumerate(entries)
    ]
    return opening, ledger


def test_rules_only_suite_is_deterministic_and_exclusion_screened() -> None:
    empty = {"canonical": set(), "float32": set()}
    first, _ = protocol.make_suite(empty)
    second, _ = protocol.make_suite(empty)
    assert protocol.suite_bytes(first) == protocol.suite_bytes(second)
    assert len(first) == 1024
    assert len({row["state_hash"] for row in first}) == 1024
    assert all(4 <= len(row["prefix_moves"]) <= 12 for row in first)


def test_command_routes_fixed_outputs_and_preserves_runtime_contract(
    tmp_path: Path,
) -> None:
    out = tmp_path / protocol.OUT
    out.mkdir(parents=True)
    (out / "openings.jsonl").write_bytes(protocol.canonical_json(_opening()) + b"\n")
    (out / "registration.json").write_text(
        json.dumps(
            {
                "runtime": {"search_options": {"fpu_mode": "zero"}},
                "runtime_assets": {
                    "exact_root_native_probe": {"artifact": "native-probe"},
                    "exact_root_tablebase": {
                        "artifact": "tablebase",
                        "runtime_materialized_path": "/tmp/tablebase",
                    },
                },
            }
        )
    )
    argv = protocol.command(tmp_path, 0)
    for flag, suffix in (
        ("--challenger", "/E4"),
        ("--challenger-policy-artifact", "/E4"),
        ("--challenger-value-artifact", "/initializer"),
        ("--current", "/initializer"),
    ):
        assert argv[argv.index(flag) + 1].endswith(suffix)
    for flag, expected in (
        ("--challenger-simulations", "128"),
        ("--current-simulations", "128"),
        ("--workers", "1"),
        ("--c-puct", "1.25"),
        ("--seed", "511"),
        ("--max-moves", "200"),
        ("--games", "2"),
    ):
        assert argv[argv.index(flag) + 1] == expected


def test_interruption_resume_skips_verified_pair_and_preserves_partial_batch(
    tmp_path: Path,
) -> None:
    opening, ledger = _pair(tmp_path)
    remaining = protocol.resume_plan(
        tmp_path, [opening, {**opening, "opening_index": 1}], ledger
    )
    assert remaining == [1]
    interrupted = tmp_path / protocol.OUT / "batches/games-0001.jsonl"
    interrupted.write_text("synthetic interrupted output\n")
    with pytest.raises(ValueError, match="seed511_interrupted_pair_preserved:1"):
        protocol.resume_plan(
            tmp_path, [opening, {**opening, "opening_index": 1}], ledger
        )


def test_semantic_runtime_corruption_is_rejected_even_after_receipt_rebinding(
    tmp_path: Path,
) -> None:
    opening, ledger = _pair(tmp_path)
    report_path = tmp_path / protocol.OUT / "batches/report-0000.json"
    report = json.loads(report_path.read_text())
    report["notes"]["challenger_simulations"] = 129
    report_path.write_text(json.dumps(report))
    ledger[0]["batch_report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    ledger[1]["batch_report_sha256"] = ledger[0]["batch_report_sha256"]
    with pytest.raises(ValueError, match="pair_runtime:0"):
        verifier.verify_pair(tmp_path, 0, opening, ledger)
