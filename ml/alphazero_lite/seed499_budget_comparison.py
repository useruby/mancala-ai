"""Frozen, resumable seed499 128-simulation comparison against seed488."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite.seed499_budget_analysis import analyze, load_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed499-budget-comparison")
HISTORICAL = Path("docs/data/seed488-frozen-strength-evaluation")
SIMULATIONS = 128
SEED = 488
ANALYSIS_SEED = 499
GAME_COUNT = 1024
SOURCE_FILES = (
    "ml/alphazero_lite/seed499_budget_comparison.py",
    "ml/alphazero_lite/verify_seed499_budget_comparison.py",
    "ml/alphazero_lite/seed499_budget_analysis.py",
    "ml/alphazero_lite/test_seed499_budget_comparison.py",
    "ml/alphazero_lite/test_seed499_budget_analysis.py",
    "ml/alphazero_lite/test_seed488_strength_evaluation.py",
    "ml/alphazero_lite/test_seed488_opening_replay.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(root: Path = ROOT) -> dict[str, Any]:
    """Create a separate seed499 registration and pre-execution freeze."""
    root = root.resolve()
    out, historical = root / OUT, root / HISTORICAL
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed499_output_exists")
    out.mkdir(parents=True, exist_ok=True)
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(root),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if json.loads(verified.stdout).get("status") != "verified_complete":
        raise ValueError("historical_seed488_verification")
    base_registration = _json(historical / "registration.json")
    for filename in ("openings.jsonl",):
        shutil.copyfile(historical / filename, out / filename)
    shutil.copytree(historical / "runtime-artifacts", out / "runtime-artifacts")
    shutil.copyfile(
        historical / "runtime-artifacts.json", out / "runtime-artifacts.json"
    )
    registration = {
        "schema": "seed499-budget-comparison-v1",
        "status": "frozen_before_execution",
        "interpretation": "exploratory budget-sensitivity analysis; not fresh confirmation",
        "fresh_confirmation_authorized": False,
        "promotion": False,
        "models": {
            "same_registered_seed488_models": True,
            "seed488_registration_sha256": _sha(historical / "registration.json"),
            "seed488_runtime_artifacts_sha256": _sha(
                historical / "runtime-artifacts.json"
            ),
            "seed488_outcomes_sha256": _sha(historical / "outcomes.jsonl"),
            "seed488_results_sha256": _sha(historical / "results.json"),
            "seed488_analysis_sha256": _sha(historical / "analysis.json"),
        },
        "opening_suite": {
            "path": "../seed488-frozen-strength-evaluation/openings.jsonl",
            "sha256": _sha(out / "openings.jsonl"),
            "count": 512,
            "same_openings_and_both_seats": True,
            "seed_derivation": "488 + parent_opening_index",
        },
        "runtime": {
            "identities_sha256": base_registration["runtime"]["identities_sha256"],
            "python_version": base_registration["runtime"]["python_version"],
            "numpy_version": base_registration["runtime"]["numpy_version"],
            "simulations_per_side": SIMULATIONS,
            "c_puct": 1.25,
            "workers": 1,
            "max_moves": 200,
            "search_options": base_registration["runtime"]["search_options"],
            "exact_root": base_registration["runtime"]["exact_root"],
            "seed_contract": "azlite_eval_seed_v2; base seed = 488 + opening index",
        },
        "games": GAME_COUNT,
        "decision_rule": "Supports budget sensitivity only if the one-sided 95% lower percentile bound of the matched 128 minus historical 1,200 opening-cluster contrast exceeds zero.",
        "standalone_interpretation": "A positive budget contrast alone does not demonstrate strength above the initializer.",
        "analysis": {
            "replicates": 20_000,
            "seed": ANALYSIS_SEED,
            "unit": "512 matched opening clusters",
            "method": "percentile bootstrap; resample matched opening differences with replacement",
        },
        "historical_comparison": "Seed488's completed 1,200-simulation publication is reused only as historical matched evidence.",
        "overlap_limitation": "Non-overlap with the unresolved seed461 continuation population cannot be certified.",
        "no_training": True,
        "source_sha256": {path: _sha(root / path) for path in SOURCE_FILES},
    }
    _write_json(out / "registration.json", registration)
    freeze = {
        "schema": "seed499-pre-execution-freeze-v1",
        "status": "frozen_before_execution",
        "registration_sha256": _sha(out / "registration.json"),
        "opening_suite_sha256": _sha(out / "openings.jsonl"),
        "runtime_artifacts_sha256": _sha(out / "runtime-artifacts.json"),
        "source_sha256": {path: _sha(root / path) for path in SOURCE_FILES},
        "historical_seed488_publication_sha256": _sha(
            historical / "publication-receipt.json"
        ),
        "historical_seed488_outcomes_sha256": _sha(historical / "outcomes.jsonl"),
    }
    _write_json(out / "freeze.json", freeze)
    _write_json(out / "receipt.json", {"status": "seed499_frozen_not_executed"})
    return freeze


def _verify_runtime_artifacts(root: Path, out: Path) -> None:
    expected = _json(root / HISTORICAL / "runtime-artifacts.json")
    actual = _json(out / "runtime-artifacts.json")
    if expected != actual:
        raise ValueError("seed499_runtime_artifact_manifest_mismatch")
    for arm, files in expected.items():
        for name, digest in files.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"seed499_runtime_artifact_hash:{arm}/{name}")


def _replay(opening: dict[str, Any], game: dict[str, Any], seat: int) -> None:
    from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

    _replay_game(opening, game, seat)


def _validate_batch(root: Path, index: int, rows: list[dict[str, Any]]) -> None:
    out = root / OUT
    openings = load_rows(out / "openings.jsonl")
    opening = openings[index]
    pair = [row for row in rows if int(row["opening_index"]) == index]
    if len(pair) != 2 or {int(row["F_player"]) for row in pair} != {0, 1}:
        raise ValueError(f"committed_pair_coverage:{index}")
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    game_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    if opening_path.read_bytes() != base.canonical_json(opening) + b"\n":
        raise ValueError(f"committed_opening_bytes:{index}")
    games = load_rows(game_path)
    report = _json(report_path)
    if len(games) != 2 or {int(row["challenger_player"]) for row in games} != {0, 1}:
        raise ValueError(f"committed_games_coverage:{index}")
    if report.get("games_played") != 2:
        raise ValueError(f"committed_report_games:{index}")
    notes = report.get("notes", {})
    if (
        notes.get("challenger_simulations") != SIMULATIONS
        or notes.get("current_simulations") != SIMULATIONS
        or notes.get("workers_used") != 1
        or notes.get("base_seed") != SEED + index
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("search_profile", {}).get("c_puct") != 1.25
    ):
        raise ValueError(f"committed_runtime_contract:{index}")
    for outcome in pair:
        pos = int(outcome["batch_game_position"])
        game = games[pos]
        if (
            outcome.get("batch_game_sha256") != _sha(game_path)
            or outcome.get("batch_report_sha256") != _sha(report_path)
            or int(game["challenger_player"]) != int(outcome["F_player"])
            or outcome.get("opening_state_hash") != opening["state_hash"]
        ):
            raise ValueError(f"committed_game_binding:{index}")
        _replay(opening, game, int(outcome["F_player"]))


def _arena_command(root: Path, index: int, opening: dict[str, Any]) -> list[str]:
    out = root / OUT
    batch = out / "batches"
    opener = batch / f"opening-{index:04d}.jsonl"
    opener.parent.mkdir(parents=True, exist_ok=True)
    payload = base.canonical_json(opening) + b"\n"
    if opener.exists() and opener.read_bytes() != payload:
        raise ValueError(f"batch_opening_conflict:{index}")
    if not opener.exists():
        opener.write_bytes(payload)
    historical_registration = _json(root / HISTORICAL / "registration.json")
    identity = base.batch_opening_identity(historical_registration, opening, opener)
    # The established seed488 command builder supplies every frozen arena option;
    # only the two registered simulation counts are lowered for this protocol.
    command = base.build_arena_command(
        historical_registration,
        out / "runtime-artifacts",
        opener,
        batch / f"report-{index:04d}.json",
        batch / f"games-{index:04d}.jsonl",
        identity,
    )
    for flag in ("--challenger-simulations", "--current-simulations"):
        command[command.index(flag) + 1] = str(SIMULATIONS)
    return command


def run(root: Path = ROOT, *, execute: bool = False) -> dict[str, Any] | None:
    root = root.resolve()
    out = root / OUT
    registration = _json(out / "registration.json")
    freeze = _json(out / "freeze.json")
    if freeze.get("status") != "frozen_before_execution":
        raise ValueError("seed499_not_frozen")
    if _sha(out / "registration.json") != freeze.get("registration_sha256"):
        raise ValueError("seed499_registration_identity")
    if not execute:
        raise ValueError("execution_requires_explicit_--execute")
    for path, digest in registration["source_sha256"].items():
        if _sha(root / path) != digest:
            raise ValueError(f"frozen_source_changed:{path}")
    for path, digest in registration["runtime"]["identities_sha256"].items():
        if _sha(root / path) != digest:
            raise ValueError(f"runtime_identity:{path}")
    _verify_runtime_artifacts(root, out)
    ledger = out / "outcomes.jsonl"
    pending = out / "outcomes.jsonl.pending"
    if pending.exists():
        raise ValueError("pending_outcome_commit_requires_explicit_recovery")
    rows = load_rows(ledger) if ledger.exists() else []
    openings = load_rows(out / "openings.jsonl")
    seen: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seen.setdefault(int(row["opening_index"]), []).append(row)
    if any(len(pair) != 2 for pair in seen.values()):
        raise ValueError("partial_committed_pair")
    for index, pair in seen.items():
        _validate_batch(root, index, rows)
    batch_dir = out / "batches"
    batch_dir.mkdir(exist_ok=True)
    for opening in openings:
        index = int(opening["opening_index"])
        if index in seen:
            continue
        game_path = batch_dir / f"games-{index:04d}.jsonl"
        report_path = batch_dir / f"report-{index:04d}.json"
        if game_path.exists() or report_path.exists():
            raise ValueError(f"interrupted_outcome_batch_preserved:{index}")
        subprocess.run(_arena_command(root, index, opening), cwd=root, check=True)
        games = load_rows(game_path)
        if len(games) != 2 or {int(game["challenger_player"]) for game in games} != {
            0,
            1,
        }:
            raise ValueError(f"arena_pair_incomplete_preserved:{index}")
        normalized = []
        for position, game in enumerate(games):
            seat = int(game["challenger_player"])
            _replay(opening, game, seat)
            normalized.append(
                {
                    **game,
                    "opening_index": index,
                    "opening_state_hash": opening["state_hash"],
                    "F_player": seat,
                    "batch_game_position": position,
                    "batch_game_sha256": _sha(game_path),
                    "batch_report_sha256": _sha(report_path),
                    "batch_opening_sha256": _sha(
                        batch_dir / f"opening-{index:04d}.jsonl"
                    ),
                }
            )
        _validate_batch(root, index, [*rows, *normalized])
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized)
        pending.write_text(ledger.read_text() + payload if ledger.exists() else payload)
        pending.replace(ledger)
        rows.extend(normalized)
    if len(rows) != GAME_COUNT:
        raise ValueError("final_game_count")
    historical = load_rows(root / HISTORICAL / "outcomes.jsonl")
    result = analyze(rows, historical)
    _write_json(out / "analysis.json", result)
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "publication-receipt.json":
            files[str(path.relative_to(out))] = _sha(path)
    _write_json(
        out / "publication-receipt.json",
        {"status": "seed499_complete", "files": files},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root) if args.prepare else run(args.root, execute=True)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
