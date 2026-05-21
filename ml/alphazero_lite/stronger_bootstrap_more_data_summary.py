from __future__ import annotations

import json
from pathlib import Path


REQUIRED_ARTIFACT_FILES = (
    "weights.json",
    "metadata.json",
    "arena_report.json",
    "run_manifest.json",
)
REQUIRED_REPORT_FILES = (
    "local_promotion_gate.json",
    "candidate_vs_current_arena.json",
    "candidate_vs_mcts1200.json",
    "current_vs_mcts1200.json",
)


def load_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def detect_candidate_dir(downloaded_root: Path) -> Path | None:
    candidates = []
    for child in sorted(downloaded_root.iterdir()) if downloaded_root.exists() else []:
        if child.is_dir() and all(
            (child / name).is_file() for name in REQUIRED_ARTIFACT_FILES
        ):
            candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    return None


def arena_score(report: dict | None) -> float | None:
    if not isinstance(report, dict):
        return None
    games = int(report.get("games_played", 0))
    if games <= 0:
        return None
    wins = int(report.get("wins", 0))
    draws = int(report.get("draws", 0))
    return round((wins + (0.5 * draws)) / games, 4)


def mcts_score(report: dict | None) -> float | None:
    if not isinstance(report, dict):
        return None
    games = int(report.get("games", 0))
    if games <= 0:
        return None
    wins = int(report.get("az_wins", 0))
    draws = int(report.get("draws", 0))
    return round((wins + (0.5 * draws)) / games, 4)


def summarize_downloaded_results(
    *,
    local_results_path: str,
    results_path: str,
    experiment_report_path: str | None,
    experiment_passed: bool | None,
    manifest_path: str | None,
    manifest_status: str | None,
) -> dict:
    downloaded_root = Path(local_results_path) / Path(results_path).name
    report_paths = {name: downloaded_root / name for name in REQUIRED_REPORT_FILES}
    gate_report = load_json(report_paths["local_promotion_gate.json"])
    arena_report = load_json(report_paths["candidate_vs_current_arena.json"])
    candidate_mcts_report = load_json(report_paths["candidate_vs_mcts1200.json"])
    current_mcts_report = load_json(report_paths["current_vs_mcts1200.json"])
    regression_path = downloaded_root / "candidate_regression_suite.json"
    forensic_path = downloaded_root / "candidate_forensic_suite.json"
    candidate_dir = detect_candidate_dir(downloaded_root)

    completed = (
        candidate_dir is not None
        and gate_report is not None
        and all(path.is_file() for path in report_paths.values())
        and regression_path.is_file()
    )
    recommendation = None
    recommendation_reason = None

    computed_arena_score = arena_score(arena_report)
    computed_candidate_mcts_score = mcts_score(candidate_mcts_report)
    computed_current_mcts_score = mcts_score(current_mcts_report)

    if completed:
        gate_passed = gate_report.get("passed") is True
        if gate_passed:
            recommendation = "confirm"
            recommendation_reason = "candidate passed the local gate, but alternate-seed confirmation is still required"
        elif (
            computed_arena_score is not None
            and computed_arena_score > 0.5
            and computed_candidate_mcts_score is not None
            and computed_current_mcts_score is not None
            and computed_candidate_mcts_score < computed_current_mcts_score
        ):
            recommendation = "pivot"
            recommendation_reason = (
                "candidate beat current in arena play but regressed against MCTS1200"
            )
        else:
            recommendation = "reject"
            recommendation_reason = (
                "candidate did not satisfy the promotion-readiness decision rules"
            )

    return {
        "results_path": results_path,
        "local_results_path": local_results_path,
        "downloaded_results_root": str(downloaded_root),
        "summary_path": str(downloaded_root / "issue1_summary.json"),
        "candidate_path": str(candidate_dir) if candidate_dir is not None else None,
        "local_promotion_gate_path": str(report_paths["local_promotion_gate.json"])
        if report_paths["local_promotion_gate.json"].is_file()
        else None,
        "candidate_vs_current_arena_path": str(
            report_paths["candidate_vs_current_arena.json"]
        )
        if report_paths["candidate_vs_current_arena.json"].is_file()
        else None,
        "candidate_vs_mcts1200_path": str(report_paths["candidate_vs_mcts1200.json"])
        if report_paths["candidate_vs_mcts1200.json"].is_file()
        else None,
        "current_vs_mcts1200_path": str(report_paths["current_vs_mcts1200.json"])
        if report_paths["current_vs_mcts1200.json"].is_file()
        else None,
        "candidate_regression_suite_path": str(regression_path)
        if regression_path.is_file()
        else None,
        "candidate_forensic_suite_path": str(forensic_path)
        if forensic_path.is_file()
        else None,
        "experiment_report_path": experiment_report_path,
        "experiment_passed": experiment_passed,
        "manifest_path": manifest_path,
        "manifest_status": manifest_status,
        "completed": completed,
        "passed": gate_report.get("passed") if isinstance(gate_report, dict) else None,
        "arena_score": gate_report.get("arena_score")
        if isinstance(gate_report, dict)
        else computed_arena_score,
        "candidate_mcts_score": gate_report.get("candidate_mcts_score")
        if isinstance(gate_report, dict)
        else computed_candidate_mcts_score,
        "current_mcts_score": gate_report.get("current_mcts_score")
        if isinstance(gate_report, dict)
        else computed_current_mcts_score,
        "failure_reasons": gate_report.get("failure_reasons")
        if isinstance(gate_report, dict)
        else None,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }


def write_summary(summary: dict) -> dict:
    summary_path = Path(summary["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
