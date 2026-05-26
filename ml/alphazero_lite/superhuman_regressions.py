from __future__ import annotations

import json
from pathlib import Path

from ml.alphazero_lite import arena

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
) -> dict:
    effective_simulations = (
        DEFAULT_SIMULATIONS if simulations is None else int(simulations)
    )
    summary = arena.evaluate_artifact_position(
        artifact_path=artifact_path,
        state=position["state"],
        simulations=effective_simulations,
        seed=seed,
        c_puct=c_puct,
        search_options=build_search_options()
        if search_options is None
        else dict(search_options),
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
) -> list[dict]:
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


def _metadata_for(result: dict) -> tuple[str, int, tuple[int, ...]]:
    return (
        str(result.get("description", "")),
        int(result["expected_move"]),
        tuple(int(move) for move in result.get("acceptable_moves", [])),
        str(_optional_semantic_metadata(result)["token"]),
        _optional_semantic_metadata(result)["move_number"],
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
