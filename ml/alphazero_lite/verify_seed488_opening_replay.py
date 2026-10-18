"""Read-only recovery verification for seed488's preserved opening-0 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed488-frozen-strength-evaluation")
ATTEMPT = "resume-corrected"
SEED = 488


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def replay_game(
    opening: dict[str, Any], row: dict[str, Any], seat: int
) -> dict[str, Any]:
    """Replay the arena's absolute pit-index trajectory from its frozen opening."""
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list):
        raise ValueError("opening_prefix_missing")
    for relative in prefix:
        if game.over() or int(relative) not in game.possible_moves():
            raise ValueError("opening_prefix_replay")
        if not game.move(game.pit_index(int(relative))):
            raise ValueError("opening_prefix_replay")
    if game.to_state() != opening.get("state"):
        raise ValueError("opening_prefix_state")

    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("trajectory_missing")
    absolute_moves = [int(value) for value in trajectory.split(",") if value]
    if len(absolute_moves) > 200 or int(row.get("game_length", -1)) != len(
        absolute_moves
    ):
        raise ValueError("trajectory_length_or_cap")
    first_move_by_player: dict[int, int] = {}
    for ply, absolute in enumerate(absolute_moves):
        relative = absolute - game.current_player * 6
        if (
            game.over()
            or not 0 <= absolute < 12
            or relative not in game.possible_moves()
            or absolute // 6 != game.current_player
        ):
            raise ValueError(f"trajectory_illegal:{ply}")
        acting_player = game.current_player
        first_move_by_player.setdefault(acting_player, relative)
        if not game.move(absolute):
            raise ValueError(f"trajectory_replay:{ply}")
    if not (game.over() and len(absolute_moves) < 200) and len(absolute_moves) != 200:
        raise ValueError("trajectory_ended_before_terminal_or_cap")

    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("winner") != winner or int(row.get("margin", 10**9)) != margin:
        raise ValueError("trajectory_outcome_mismatch")
    if row.get("first_move_challenger") != first_move_by_player.get(seat):
        raise ValueError("first_challenger_move_mismatch")
    if row.get("first_move_current") != first_move_by_player.get(1 - seat):
        raise ValueError("first_current_move_mismatch")
    return {"winner": winner, "margin": margin, "game_length": len(absolute_moves)}


def _preserved_snapshot(out: Path) -> dict[Path, tuple[bytes, int]]:
    provenance_path = out / "failure-provenance.json"
    provenance = _json(provenance_path)
    paths = {out / "execution-freeze.json", provenance_path}
    paths.update(out / name for name in provenance["preserved_files"])
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def _assert_snapshot(snapshot: dict[Path, tuple[bytes, int]]) -> None:
    for path, (content, mtime_ns) in snapshot.items():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"preserved_evidence_changed:{path.name}")
        if path.stat().st_mtime_ns != mtime_ns:
            raise ValueError(f"preserved_evidence_mtime_changed:{path.name}")


def _verify(root: Path, *, allow_prospective_sources: bool = False) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    receipt_path = out / "receipt.json"
    suite_path = out / "openings.jsonl"
    freeze_path = out / "execution-freeze.json"
    provenance_path = out / "failure-provenance.json"
    registration = _json(registration_path)
    receipt = _json(receipt_path)
    freeze = _json(freeze_path)
    provenance = _json(provenance_path)

    if (
        freeze.get("schema") != "seed488-pre-game-execution-freeze-v1"
        or freeze.get("status") != "frozen_before_execution_review_required"
        or freeze.get("execution_namespace") != ATTEMPT
        or freeze.get("registration_sha256") != _sha(registration_path)
        or freeze.get("receipt_sha256") != _sha(receipt_path)
        or freeze.get("opening_suite_sha256") != _sha(suite_path)
        or freeze.get("failure_provenance_sha256") != _sha(provenance_path)
    ):
        raise ValueError("execution_freeze_binding")
    for relative, digest in receipt.items():
        if relative != "status" and _sha(out / relative) != digest:
            raise ValueError(f"preparation_receipt_mismatch:{relative}")
    for relative, digest in freeze.get("source_sha256", {}).items():
        if _sha(root / relative) == digest:
            continue
        if allow_prospective_sources and relative in {
            "ml/alphazero_lite/seed488_strength_evaluation.py",
            "ml/alphazero_lite/test_seed488_strength_evaluation.py",
            "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
        }:
            continue
        recovery_path = out / "opening-0000-recovery-receipt.json"
        if not recovery_path.is_file():
            raise ValueError(f"frozen_source_binding:{relative}")
        recovery = _json(recovery_path)
        if relative == "ml/alphazero_lite/test_seed488_strength_evaluation.py":
            if recovery.get("historical_execution", {}).get(
                "source_sha256"
            ) != freeze.get("source_sha256"):
                raise ValueError(f"historical_source_binding:{relative}")
            continue
        prospective = recovery.get("prospective_correction", {}).get(
            "source_sha256", {}
        )
        actual = _sha(root / relative)
        source_bound = prospective.get(relative) == actual
        continuation_path = out / "opening-0000-continuation-binding-v3.json"
        if not source_bound and continuation_path.is_file():
            continuation = _json(continuation_path)
            source_bound = (
                continuation.get("recovery_receipt_sha256") == _sha(recovery_path)
                and continuation.get("source_sha256", {}).get(relative) == actual
            )
        if (
            recovery.get("historical_execution", {}).get("source_sha256")
            != freeze.get("source_sha256")
            or not source_bound
        ):
            raise ValueError(f"prospective_source_binding:{relative}")
    if (
        provenance.get("status") != "seed488_attempt_491_failed_before_game"
        or provenance.get("completed_games") != 0
        or provenance.get("outcomes_committed") is not False
    ):
        raise ValueError("failure_provenance_status")
    for relative, digest in provenance.get("preserved_files", {}).items():
        if _sha(out / relative) != digest:
            raise ValueError(f"failed_attempt_evidence_binding:{relative}")

    expected_runtime = registration["runtime"]["identities_sha256"]
    if registration["runtime"].get("python_version") != sys.version:
        raise ValueError("python_runtime_version")
    import numpy as np

    if registration["runtime"].get("numpy_version") != np.__version__:
        raise ValueError("numpy_runtime_version")
    for relative, digest in expected_runtime.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"runtime_binding:{relative}")
    manifest_path = out / "runtime-artifacts.json"
    if _sha(manifest_path) != freeze.get("runtime_artifact_manifest_sha256"):
        raise ValueError("artifact_manifest_binding")
    manifest = _json(manifest_path)
    if manifest != freeze.get("runtime_artifacts_sha256"):
        raise ValueError("artifact_manifest_identity")
    for arm, files in manifest.items():
        for name, digest in files.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"artifact_binding:{arm}:{name}")

    suite_rows = _rows(suite_path)
    if len(suite_rows) != 512 or [row["opening_index"] for row in suite_rows] != list(
        range(512)
    ):
        raise ValueError("parent_suite_coverage")
    opening = suite_rows[0]
    batch_out = out / ATTEMPT
    opening_path = batch_out / "opening-0000.jsonl"
    expected_opening = (
        json.dumps(opening, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if opening_path.read_bytes() != expected_opening:
        raise ValueError("batch_parent_opening_identity")
    batch_digest = _sha(opening_path)
    if batch_digest != freeze.get("failed_batch_sha256"):
        raise ValueError("corrected_opening_batch_identity")
    game_path = batch_out / "games-0000.jsonl"
    report_path = batch_out / "report-0000.json"
    games = _rows(game_path)
    report = _json(report_path)
    if len(games) != 2 or {int(row["challenger_player"]) for row in games} != {0, 1}:
        raise ValueError("seat_pair_coverage")
    if report.get("games_played") != 2:
        raise ValueError("actual_report_game_count")
    notes = report.get("notes", {})
    expected_options = dict(registration["runtime"]["search_options"])
    expected_options.pop("random_opening_plies", None)
    expected_options.pop("root_prior_transform", None)
    checks = {
        "challenger_simulations": 1200,
        "current_simulations": 1200,
        "workers_used": 1,
        "random_opening_plies": 0,
        "seed_contract": "azlite_eval_seed_v2",
        "base_seed": SEED,
        "seed": SEED,
        "suite_sha256": batch_digest,
        "search_options": expected_options,
        "runtime_search_policy_sha256": expected_runtime[
            "model-artifact/current/search_policy.json"
        ],
        "exact_root_native_probe_sha256": expected_runtime[
            "model-artifact/runtime/kalah_v1_tablebase"
        ],
        "exact_root_tablebase_sha256": expected_runtime[
            "model-artifact/runtime/kalah_v1_21.kvtb"
        ],
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }
    if notes.get("search_profile", {}).get("c_puct") != 1.25:
        raise ValueError("actual_report_c_puct")
    for name, value in checks.items():
        if notes.get(name) != value:
            raise ValueError(f"actual_report_{name}")

    verified = []
    for row in games:
        seat = int(row["challenger_player"])
        if (
            int(row.get("opening_index", -1)) != 0
            or int(row.get("game_within_opening", -1)) not in (0, 1)
            or row.get("opening_state_hash") != opening["state_hash"]
            or row.get("opening_prefix_moves") != opening["prefix_moves"]
            or row.get("opening_contract") != "arena_player_relative_v2"
            or int(row.get("opening_applied_prefix_length", -1))
            != len(opening["prefix_moves"])
        ):
            raise ValueError("arena_game_parent_identity")
        verified.append({"challenger_player": seat, **replay_game(opening, row, seat)})

    if report.get("wins") != sum(row["winner"] == "challenger" for row in games):
        raise ValueError("actual_report_wins")
    if report.get("losses") != sum(row["winner"] == "current" for row in games):
        raise ValueError("actual_report_losses")
    if report.get("draws") != sum(row["winner"] == "draw" for row in games):
        raise ValueError("actual_report_draws")
    return {
        "status": "opening_0000_pair_replay_verified",
        "opening_index": 0,
        "parent_suite_sha256": _sha(suite_path),
        "batch_suite_sha256": batch_digest,
        "game_log_sha256": _sha(game_path),
        "report_sha256": _sha(report_path),
        "games": sorted(verified, key=lambda item: item["challenger_player"]),
    }


def verify(root: Path) -> dict[str, Any]:
    out = root.resolve() / OUT
    snapshot = _preserved_snapshot(out)
    try:
        return _verify(root)
    finally:
        _assert_snapshot(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
