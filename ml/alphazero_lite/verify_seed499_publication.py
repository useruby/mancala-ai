"""Independent read-only verification for the seed499 budget comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed499_budget_analysis import analyze, load_rows
from ml.alphazero_lite.seed499_budget_comparison import (
    GAME_COUNT,
    HISTORICAL,
    OUT,
    SOURCE_FILES,
    _sha,
)
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    if registration.get("schema") != "seed499-budget-comparison-v1":
        raise ValueError("registration_schema")
    if freeze.get("schema") != "seed499-pre-execution-freeze-v1":
        raise ValueError("freeze_schema")
    if freeze.get("status") != "frozen_before_execution":
        raise ValueError("freeze_status")
    if _sha(out / "registration.json") != freeze.get("registration_sha256"):
        raise ValueError("registration_binding")
    if _sha(out / "openings.jsonl") != registration["opening_suite"]["sha256"]:
        raise ValueError("opening_suite_binding")
    if _sha(out / "openings.jsonl") != freeze.get("opening_suite_sha256"):
        raise ValueError("freeze_opening_binding")
    if _sha(out / "runtime-artifacts.json") != freeze.get("runtime_artifacts_sha256"):
        raise ValueError("runtime_artifact_manifest_binding")
    for path in SOURCE_FILES:
        digest = registration.get("source_sha256", {}).get(path)
        if not digest or _sha(root / path) != digest:
            raise ValueError(f"source_binding:{path}")
        if freeze.get("source_sha256", {}).get(path) != digest:
            raise ValueError(f"freeze_source_binding:{path}")
    if (
        _sha(root / HISTORICAL / "registration.json")
        != registration["models"]["seed488_registration_sha256"]
    ):
        raise ValueError("historical_registration_binding")
    if (
        _sha(root / HISTORICAL / "runtime-artifacts.json")
        != registration["models"]["seed488_runtime_artifacts_sha256"]
    ):
        raise ValueError("historical_artifact_manifest_binding")
    for filename, field in (
        ("outcomes.jsonl", "seed488_outcomes_sha256"),
        ("results.json", "seed488_results_sha256"),
        ("analysis.json", "seed488_analysis_sha256"),
    ):
        if _sha(root / HISTORICAL / filename) != registration["models"][field]:
            raise ValueError(f"historical_seed488_binding:{filename}")
    if _sha(root / HISTORICAL / "publication-receipt.json") != freeze.get(
        "historical_seed488_publication_sha256"
    ):
        raise ValueError("historical_publication_receipt_binding")
    if _sha(root / HISTORICAL / "outcomes.jsonl") != freeze.get(
        "historical_seed488_outcomes_sha256"
    ):
        raise ValueError("historical_outcomes_freeze_binding")
    expected_artifacts = json.loads(
        (root / HISTORICAL / "runtime-artifacts.json").read_text()
    )
    actual_artifacts = json.loads((out / "runtime-artifacts.json").read_text())
    if expected_artifacts != actual_artifacts:
        raise ValueError("runtime_artifact_manifest_mismatch")
    for arm, items in expected_artifacts.items():
        for name, digest in items.items():
            path = out / "runtime-artifacts" / arm / name
            if _sha(path) != digest:
                raise ValueError(f"runtime_artifact_hash:{arm}/{name}")
    runtime = registration["runtime"]
    if runtime.get("simulations_per_side") != 128:
        raise ValueError("simulation_budget")
    for path, digest in runtime["identities_sha256"].items():
        if _sha(root / path) != digest:
            raise ValueError(f"runtime_identity:{path}")

    openings = load_rows(out / "openings.jsonl")
    rows = load_rows(out / "outcomes.jsonl")
    if len(openings) != 512 or len(rows) != GAME_COUNT:
        raise ValueError("final_counts")
    if [int(row["opening_index"]) for row in openings] != list(range(512)):
        raise ValueError("opening_order")
    by_opening: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_opening.setdefault(int(row["opening_index"]), []).append(row)
    if set(by_opening) != set(range(512)):
        raise ValueError("outcome_opening_coverage")
    batch_dir = out / "batches"
    for index, opening in enumerate(openings):
        pair = by_opening[index]
        if len(pair) != 2 or {int(row["F_player"]) for row in pair} != {0, 1}:
            raise ValueError(f"outcome_pair_coverage:{index}")
        if len({row["opening_state_hash"] for row in pair}) != 1:
            raise ValueError(f"outcome_opening_hash:{index}")
        if pair[0]["opening_state_hash"] != opening["state_hash"]:
            raise ValueError(f"outcome_opening_state:{index}")
        opening_path = batch_dir / f"opening-{index:04d}.jsonl"
        games_path = batch_dir / f"games-{index:04d}.jsonl"
        report_path = batch_dir / f"report-{index:04d}.json"
        for row in pair:
            if _sha(games_path) != row.get("batch_game_sha256"):
                raise ValueError(f"game_log_hash:{index}")
            if _sha(report_path) != row.get("batch_report_sha256"):
                raise ValueError(f"report_hash:{index}")
            if _sha(opening_path) != row.get("batch_opening_sha256"):
                raise ValueError(f"batch_opening_hash:{index}")
        games = load_rows(games_path)
        report = json.loads(report_path.read_text())
        notes = report.get("notes", {})
        if (
            len(games) != 2
            or report.get("games_played") != 2
            or notes.get("challenger_simulations") != 128
            or notes.get("current_simulations") != 128
            or notes.get("workers_used") != 1
            or notes.get("base_seed") != 488 + index
            or notes.get("seed_contract") != "azlite_eval_seed_v2"
            or notes.get("search_profile", {}).get("c_puct") != 1.25
        ):
            raise ValueError(f"batch_runtime_contract:{index}")
        if not opening_path.is_file() or _sha(opening_path) != pair[0].get(
            "batch_opening_sha256"
        ):
            raise ValueError(f"batch_opening_hash:{index}")
        for row in pair:
            position = int(row["batch_game_position"])
            game = games[position]
            if int(game["challenger_player"]) != int(row["F_player"]):
                raise ValueError(f"game_seat:{index}:{position}")
            replayed = _replay_game(opening, game, int(row["F_player"]))
            if row.get("winner") != replayed[0]:
                raise ValueError(f"outcome_replay:{index}:{position}")
    historical_rows = load_rows(root / HISTORICAL / "outcomes.jsonl")
    result = analyze(rows, historical_rows)
    published = json.loads((out / "analysis.json").read_text())
    if result != published:
        raise ValueError("analysis_mismatch")
    receipt = json.loads((out / "publication-receipt.json").read_text())
    if receipt.get("status") != "seed499_complete":
        raise ValueError("publication_status")
    for relative, digest in receipt.get("files", {}).items():
        if _sha(out / relative) != digest:
            raise ValueError(f"publication_file_hash:{relative}")
    return {
        "status": "verified_complete",
        "games": len(rows),
        "opening_count": len(openings),
        "analysis": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
