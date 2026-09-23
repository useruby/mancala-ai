from __future__ import annotations

import json
from pathlib import Path

from ml.alphazero_lite import arena
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

DEFAULT_SIMULATIONS = 384


def load_regression_positions(path: str | Path) -> list[dict]:
    positions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(positions, list):
        raise ValueError("regression positions fixture must be a JSON array")
    return positions


def build_search_options(**overrides: object) -> dict:
    return arena.build_eval_search_options(**overrides)


def _optional_semantic_metadata(position_or_result: dict) -> dict:
    token = position_or_result.get("token")
    move_number = position_or_result.get("move_number")
    metadata = {
        "token": "" if token is None else str(token),
        "move_number": None if move_number is None else int(move_number),
    }
    return metadata


def evaluate_regression_position(
    *,
    position: dict,
    artifact_path: str | Path,
    simulations: int | None,
    seed: int,
    c_puct: float,
    search_options: dict | None = None,
    endgame_tablebase: object | None = None,
    exact_solve_stone_threshold: int | None = None,
    exact_root_solve_threshold: int | None = None,
) -> dict:
    effective_simulations = (
        DEFAULT_SIMULATIONS if simulations is None else int(simulations)
    )
    evaluation_kwargs = {
        "artifact_path": artifact_path,
        "state": position["state"],
        "simulations": effective_simulations,
        "seed": seed,
        "c_puct": c_puct,
        "search_options": build_search_options()
        if search_options is None
        else dict(search_options),
    }
    if exact_root_solve_threshold is not None:
        evaluation_kwargs.update(
            endgame_tablebase=endgame_tablebase,
            exact_root_solve_threshold=exact_root_solve_threshold,
        )
    summary = arena.evaluate_artifact_position(
        **evaluation_kwargs,
    )
    expected_move = int(position["expected_move"])
    acceptable_moves = [int(move) for move in position.get("acceptable_moves", [])]
    passing_moves = acceptable_moves or [expected_move]
    selected_move = summary.get("selected_move")
    passed = selected_move in passing_moves
    return {
        "id": str(position["id"]),
        "description": str(position.get("description", "")),
        "expected_move": expected_move,
        "acceptable_moves": passing_moves,
        **_optional_semantic_metadata(position),
        "selected_move": selected_move,
        "passed": bool(passed),
        "summary": summary,
    }


def evaluate_regression_positions(
    *,
    positions: list[dict],
    artifact_path: str | Path,
    simulations: int | None,
    seed: int,
    c_puct: float,
    search_options: dict | None = None,
    exact_solve_stone_threshold: int | None = None,
    exact_root_solve_threshold: int | None = None,
    exact_root_native_probe: str | Path | None = None,
    exact_root_tablebase: str | Path | None = None,
) -> list[dict]:
    if (exact_root_native_probe is None) != (exact_root_tablebase is None):
        raise ValueError(
            "exact_root_native_probe and exact_root_tablebase must be supplied together"
        )
    if exact_root_solve_threshold is not None:
        if exact_root_solve_threshold != EndgameTablebase.MAX_SOLVED_SEEDS:
            raise ValueError(
                "exact_root_solve_threshold must equal EndgameTablebase.MAX_SOLVED_SEEDS"
            )
        if exact_solve_stone_threshold is not None:
            raise ValueError(
                "exact root handoff and exact leaf solving cannot be combined"
            )
        if exact_root_native_probe is None:
            raise ValueError("exact root handoff requires native probe and tablebase")

        with NativeExactRootTablebase(
            exact_root_native_probe, exact_root_tablebase, warm_on_start=True
        ) as native_root_tablebase:
            return [
                evaluate_regression_position(
                    position=position,
                    artifact_path=artifact_path,
                    simulations=simulations,
                    seed=seed,
                    c_puct=c_puct,
                    search_options=search_options,
                    endgame_tablebase=native_root_tablebase,
                    exact_root_solve_threshold=exact_root_solve_threshold,
                )
                for position in positions
            ]

    return [
        evaluate_regression_position(
            position=position,
            artifact_path=artifact_path,
            simulations=simulations,
            seed=seed,
            c_puct=c_puct,
            search_options=search_options,
        )
        for position in positions
    ]


def build_regression_report(
    *, artifact_path: str | Path, positions_path: str | Path, results: list[dict]
) -> dict:
    return {
        "passed": bool(results)
        and all(bool(result.get("passed")) for result in results),
        "artifact_path": str(artifact_path),
        "positions_path": str(positions_path),
        "results": results,
    }


def _results_by_id(results: list[dict]) -> dict[str, dict]:
    indexed_results = {}
    for result in results:
        result_id = str(result["id"])
        if result_id in indexed_results:
            raise ValueError(f"duplicate regression result id: {result_id}")
        indexed_results[result_id] = result
    return indexed_results


def _metadata_for(result: dict) -> tuple[str, int, tuple[int, ...], str, int | None]:
    optional_metadata = _optional_semantic_metadata(result)
    return (
        str(result.get("description", "")),
        int(result["expected_move"]),
        tuple(int(move) for move in result.get("acceptable_moves", [])),
        str(optional_metadata["token"]),
        optional_metadata["move_number"],
    )


def compare_regression_results(
    *, baseline_results: list[dict], candidate_results: list[dict]
) -> list[dict]:
    baseline_by_id = _results_by_id(baseline_results)
    candidate_by_id = _results_by_id(candidate_results)
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("mismatched regression result ids")
    comparisons = []
    for baseline in baseline_results:
        candidate = candidate_by_id[str(baseline["id"])]
        if _metadata_for(baseline) != _metadata_for(candidate):
            raise ValueError(f"mismatched regression metadata for id: {baseline['id']}")
        baseline_passed = bool(baseline.get("passed"))
        candidate_passed = bool(candidate.get("passed"))
        comparisons.append(
            {
                "id": str(baseline["id"]),
                "description": str(baseline.get("description", "")),
                "expected_move": int(baseline["expected_move"]),
                "acceptable_moves": [
                    int(move) for move in baseline.get("acceptable_moves", [])
                ],
                "baseline_selected_move": baseline.get("selected_move"),
                "candidate_selected_move": candidate.get("selected_move"),
                "baseline_passed": baseline_passed,
                "candidate_passed": candidate_passed,
                "improved": (not baseline_passed) and candidate_passed,
                "regressed": baseline_passed and (not candidate_passed),
            }
        )
    return comparisons
