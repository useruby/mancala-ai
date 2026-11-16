"""Checkpointed 18-way pair-level Seed 599 arena execution."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed599_evaluation_readiness as protocol


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_exclusive(path: Path, payload: bytes) -> None:
    protocol.atomic_create(path, payload)


def assignment_key(arm: str, opening_index: int) -> str:
    return f"{arm}-opening-{opening_index:04d}"


def _command(
    root: Path,
    arm: str,
    opening: dict[str, Any],
    opening_index: int,
    batch: Path,
) -> list[str]:
    contract = protocol.read_json(root / protocol.PUBLISHED)
    settings = contract["match_settings"]
    artifacts = root / protocol.OUT / "candidate-bundles"
    serving = root / protocol.SERVING
    open_path = batch / "opening.jsonl"
    _write_exclusive(open_path, protocol.canonical(opening) + b"\n")
    policy_options = {
        "fpu_mode": settings["fpu_mode"],
        "reuse_subtree": settings["reuse_subtree"],
        "normalize_values": settings["normalize_values"],
        "root_policy_mode": settings["root_policy_mode"],
        "tactical_root_bias": settings["tactical_root_bias"],
        "root_temperature": settings["root_temperature"],
    }
    options = json.dumps(policy_options, sort_keys=True, separators=(",", ":"))
    argv = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(artifacts / arm),
        "--current",
        str(serving),
        "--challenger-policy-artifact",
        str(artifacts / arm),
        "--challenger-value-artifact",
        str(serving),
        "--current-policy-artifact",
        str(serving),
        "--current-value-artifact",
        str(serving),
        "--games",
        "2",
        "--games-per-opening",
        "2",
        "--workers",
        str(settings["workers"]),
        "--challenger-simulations",
        str(settings["simulations_per_side"]),
        "--current-simulations",
        str(settings["simulations_per_side"]),
        "--c-puct",
        str(settings["c_puct"]),
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--challenger-search-options-json",
        options,
        "--current-search-options-json",
        options,
        "--random-opening-plies",
        "0",
        "--max-moves",
        str(settings["max_plies"]),
        "--exact-root-solve-threshold",
        str(settings["exact_root_threshold"]),
        "--exact-root-native-probe",
        str((root / protocol.NATIVE_PROBE).resolve()),
        "--exact-root-tablebase",
        str((root / protocol.TABLEBASE).resolve()),
        "--seed",
        str(599 + opening_index),
        "--opening-prefixes-jsonl",
        str(open_path),
        "--suite-sha256",
        protocol.sha(open_path),
        "--game-jsonl",
        str(batch / "games.jsonl"),
        "--out",
        str(batch / "report.json"),
    ]
    return argv


def _verify_pair(
    root: Path, arm: str, opening: dict[str, Any], index: int, batch: Path
) -> dict[str, Any]:
    from ml.alphazero_lite.verify_seed599_evaluation import verify_game

    games_path, report_path = batch / "games.jsonl", batch / "report.json"
    games = _rows(games_path)
    report = protocol.read_json(report_path)
    if len(games) != 2 or {int(g.get("challenger_player", -1)) for g in games} != {
        0,
        1,
    }:
        raise ValueError(f"seat_pair_invalid:{arm}:{index}")
    for position, game in enumerate(games):
        verify_game(root, arm, opening, index, position, game)
    from ml.alphazero_lite.verify_seed599_evaluation import _check_runtime_report

    _check_runtime_report(
        root, arm, index, report, games, protocol.sha(batch / "opening.jsonl")
    )
    options = report.get("notes", {})
    exact = protocol.read_json(root / protocol.OUT / "execution-freeze.json")
    if (
        report.get("games_played") != 2
        or options.get("challenger_simulations") != 128
        or options.get("current_simulations") != 128
        or options.get("workers_used") != 1
        or options.get("base_seed") != 599 + index
        or options.get("exact_root_solve_threshold") != 16
        or options.get("exact_root_native_probe_sha256")
        != exact["checkpoint_and_serving_identities"]["exact_root_native_probe_sha256"]
        or options.get("exact_root_tablebase_sha256")
        != exact["checkpoint_and_serving_identities"]["tablebase_sha256"]
    ):
        raise ValueError(f"arena_runtime_identity_mismatch:{arm}:{index}")
    command_path = batch / "command.json"
    return {
        "schema": "seed599-evaluation-assignment-v1",
        "arm": arm,
        "opening_index": index,
        "opening_id": opening["opening_id"],
        "opening_canonical_identity": opening["canonical_identity"],
        "challenger_seats": [0, 1],
        "game_count": 2,
        "opening_file_sha256": protocol.sha(batch / "opening.jsonl"),
        "command_sha256": protocol.sha(command_path),
        "execution_environment": protocol.SEARCH_ENVIRONMENT,
        "games_sha256": protocol.sha(games_path),
        "report_sha256": protocol.sha(report_path),
        "candidate_checkpoint_sha256": protocol.read_json(
            root / protocol.OUT / "execution-freeze.json"
        )["checkpoint_and_serving_identities"]["arms"][arm]["checkpoint_sha256"],
        "candidate_runtime_model_sha256": protocol.read_json(
            root / protocol.OUT / "execution-freeze.json"
        )["candidate_bundle_manifest"][arm]["model.npz"],
        "serving_p_e4_model_sha256": protocol.read_json(
            root / protocol.OUT / "execution-freeze.json"
        )["checkpoint_and_serving_identities"]["serving_p_e4"]["model.npz"],
        "serving_p_e4_value_evaluator_sha256": protocol.read_json(
            root / protocol.OUT / "execution-freeze.json"
        )["checkpoint_and_serving_identities"]["serving_p_e4"]["model.npz"],
        "games": [
            {
                "position": pos,
                "challenger_player": int(g["challenger_player"]),
                "winner": str(g["winner"]),
                "margin": int(g["margin"]),
                "trajectory_sha256": hashlib.sha256(
                    str(g["trajectory"]).encode()
                ).hexdigest(),
            }
            for pos, g in enumerate(games)
        ],
    }


def _execute_one(root: Path, arm: str, opening: dict[str, Any], index: int) -> None:
    output = root / protocol.OUT
    name = assignment_key(arm, index)
    batch = output / "assignments" / name
    batch.mkdir(parents=True, exist_ok=True)
    evidence = batch / "assignment.json"
    existing = any(
        (batch / filename).exists()
        for filename in ("opening.jsonl", "command.json", "games.jsonl", "report.json")
    )
    if evidence.exists():
        raise ValueError(f"completed_assignment_must_be_verified_not_rerun:{name}")
    if existing:
        raise ValueError(f"incomplete_assignment_evidence_preserved:{name}")
    command = _command(root, arm, opening, index, batch)
    _write_exclusive(
        batch / "command.json",
        json.dumps(command, separators=(",", ":")).encode() + b"\n",
    )
    environment = os.environ.copy()
    environment.update(protocol.SEARCH_ENVIRONMENT)
    subprocess.run(command, cwd=root, check=True, env=environment)
    entry = _verify_pair(root, arm, opening, index, batch)
    _write_exclusive(
        evidence,
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def execute_registered(root: Path, freeze: dict[str, Any]) -> None:
    """Run only missing assignments; completed/incomplete evidence is never overwritten."""
    root = root.resolve()
    if freeze.get("execution_authorized") is not False:
        raise ValueError("freeze_must_remain_not_authorized")
    openings = protocol.read_json(root / protocol.OPENINGS)
    assignments = [
        (arm, opening, index)
        for arm in ("A", "B")
        for index, opening in enumerate(openings)
    ]
    # Reconcile completed evidence through the independent verifier before resume.
    from ml.alphazero_lite.verify_seed599_evaluation import verify_readonly

    verify_readonly(root, allow_incomplete=True)
    pending = []
    for arm, opening, index in assignments:
        evidence = (
            root
            / protocol.OUT
            / "assignments"
            / assignment_key(arm, index)
            / "assignment.json"
        )
        if evidence.exists():
            continue
        pending.append((arm, opening, index))
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        futures = [
            pool.submit(_execute_one, root, arm, opening, index)
            for arm, opening, index in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    verify_readonly(root, allow_incomplete=False)
