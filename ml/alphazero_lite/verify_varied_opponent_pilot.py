"""Read-only replay and accounting verification for the frozen P-coverage pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import run_varied_opponent_pilot as runner
from ml.alphazero_lite import varied_opponent_pilot as pilot


def _canonical_identity(state: dict[str, Any]) -> str:
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float32_identity(encoded_state: list[float]) -> str:
    encoded = np.asarray(encoded_state, dtype="<f4").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def _outcome_counts(games: list[dict[str, Any]]) -> dict[str, int]:
    outcomes: Counter[str] = Counter()
    for game in games:
        winner = game["winner"]
        seat = int(game["spec"]["designated_p_seat"])
        outcomes["draw" if winner is None else "win" if winner == seat else "loss"] += 1
    return {key: outcomes[key] for key in ("win", "draw", "loss")}


def verify(root: Path, expected_digest: str) -> dict[str, Any]:
    freeze_path = root / "docs/data/varied-opponent-coverage-pilot/freeze.json"
    freeze = runner.load_freeze(freeze_path, expected_digest=expected_digest)
    runner.verify_execution_sources(freeze, root)
    jobs = runner.pair_jobs(freeze)
    if len(jobs) != 128 or sum(len(job["game_specs"]) for job in jobs) != 256:
        raise ValueError("frozen_assignment_count_mismatch")

    base = root / "docs/data/varied-opponent-coverage-pilot"
    pair_dir = base / "pairs"
    attempt_dir = base / "attempts"
    committed, pending = pilot.load_pair_checkpoints(pair_dir, jobs)
    if pending or len(committed) != 128:
        raise ValueError("pair_checkpoint_coverage_mismatch")
    committed_by_id = {str(pair["pair_id"]): pair for pair in committed}
    if len(committed_by_id) != 128:
        raise ValueError("duplicate_committed_pair")
    pilot.reconcile_pair_attempts(attempt_dir, jobs, committed_by_id, pair_dir)
    expected_attempt_paths = {
        pilot.pair_attempt_path(attempt_dir, str(job["game_specs"][0]["pair_id"]), 1)
        for job in jobs
    }
    observed_attempt_paths = set(attempt_dir.glob("*.json"))
    if observed_attempt_paths != expected_attempt_paths:
        raise ValueError("attempt_record_set_mismatch")
    attempts = [json.loads(path.read_text()) for path in sorted(observed_attempt_paths)]
    if any(item.get("status") != "completed" for item in attempts):
        raise ValueError("attempt_not_completed")

    expected_assignments = {
        str(spec["game_id"]): spec for spec in freeze["assignments"]
    }
    games: list[dict[str, Any]] = []
    for pair in committed:
        for game in pair["games"]:
            game_id = str(game["spec"]["game_id"])
            if game_id in {str(item["spec"]["game_id"]) for item in games}:
                raise ValueError("duplicate_game_record")
            if (
                game_id not in expected_assignments
                or game["spec"] != expected_assignments[game_id]
            ):
                raise ValueError("frozen_assignment_mismatch")
            games.append(game)
    if len(games) != 256 or {str(game["spec"]["game_id"]) for game in games} != set(
        expected_assignments
    ):
        raise ValueError("game_assignment_coverage_mismatch")
    replay = pilot.verify_archive(games)
    if replay["games"] != 256:
        raise ValueError("replay_game_count_mismatch")

    accounting = pilot.summarize_pair_accounting(committed)
    accepted: dict[str, dict[str, set[str]]] = {
        scheme: {arm: set() for arm in ("control", "treatment")}
        for scheme in ("canonical_state_sha256", "kalah_v3_float32_sha256")
    }
    raw_eligible_rows: Counter[str] = Counter()
    opponent_games: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"treatment": [], "control": []}
    )
    for game in games:
        spec = game["spec"]
        opponent = str(spec["opponent"])
        opponent_games[opponent][str(spec["arm"])].append(game)
        if game["game_status"] != "complete":
            continue
        for row in game["rows"]:
            if int(row["stone_count"]) <= 32:
                continue
            arm = str(row["arm"])
            raw_eligible_rows[arm] += 1
            accepted["canonical_state_sha256"][arm].add(
                _canonical_identity(row["canonical_state"])
            )
            accepted["kalah_v3_float32_sha256"][arm].add(
                _float32_identity(row["state"])
            )

    coverage: dict[str, Any] = {}
    for scheme, arms in accepted.items():
        control, treatment = arms["control"], arms["treatment"]
        shared = control & treatment
        control_only = control - treatment
        treatment_only = treatment - control
        yield_by_arm = {
            arm: {
                "distinct_positions": len(identities),
                "cpu_hours": accounting[arm]["cpu_hours"],
                "positions_per_cpu_hour": len(identities)
                / accounting[arm]["cpu_hours"],
            }
            for arm, identities in arms.items()
        }
        control_yield = yield_by_arm["control"]["positions_per_cpu_hour"]
        treatment_yield = yield_by_arm["treatment"]["positions_per_cpu_hour"]
        coverage[scheme] = {
            "raw_eligible_rows": {
                arm: raw_eligible_rows[arm] for arm in ("control", "treatment")
            },
            "control": yield_by_arm["control"],
            "treatment": yield_by_arm["treatment"],
            "shared_positions": len(shared),
            "control_only_positions": len(control_only),
            "treatment_only_positions": len(treatment_only),
            "treatment_control_yield_ratio": treatment_yield / control_yield
            if control_yield
            else None,
        }

    matched_results: dict[str, Any] = {}
    for opponent in ("initializer", "T512"):
        by_opening: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"treatment": [], "control": []}
        )
        for game in opponent_games[opponent]["treatment"]:
            by_opening[str(game["spec"]["opening_id"])]["treatment"].append(game)
        control_by_opening: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for game in games:
            if game["spec"]["arm"] == "control":
                control_by_opening[str(game["spec"]["opening_id"])].append(game)
        rows = []
        for opening, split in sorted(by_opening.items()):
            control_games = control_by_opening[opening]
            if len(split["treatment"]) != 2 or len(control_games) != 2:
                raise ValueError("matched_opening_seat_pair_mismatch")
            rows.append(
                {
                    "opening_id": opening,
                    "treatment_p_outcomes": _outcome_counts(split["treatment"]),
                    "control_p_outcomes": _outcome_counts(control_games),
                }
            )
        if len(rows) != 32:
            raise ValueError("opponent_opening_count_mismatch")
        treatment_games = opponent_games[opponent]["treatment"]
        controls = [
            game
            for game in games
            if game["spec"]["arm"] == "control"
            and game["spec"]["opening_id"] in by_opening
        ]
        matched_results[opponent] = {
            "openings": rows,
            "treatment_p_outcomes": _outcome_counts(treatment_games),
            "same_opening_control_p_outcomes": _outcome_counts(controls),
        }

    raw_files = sorted([*pair_dir.glob("*.json"), *attempt_dir.glob("*.json")])
    raw_manifest = [
        [str(path.relative_to(base)), hashlib.sha256(path.read_bytes()).hexdigest()]
        for path in raw_files
    ]
    raw_manifest_digest = hashlib.sha256(
        json.dumps(raw_manifest, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "varied_opponent_coverage_pilot_verified_results_v1",
        "freeze_sha256": expected_digest,
        "assignments": {"games": len(games), "seat_swapped_pairs": len(committed)},
        "replay": replay,
        "accounting": accounting,
        "coverage": coverage,
        "matched_opening_opponent_results": matched_results,
        "attempts": {
            "records": len(attempts),
            "completed": len(attempts),
            "retries": 0,
        },
        "raw_evidence": {
            "pair_checkpoints": len(list(pair_dir.glob("*.json"))),
            "attempt_records": len(list(attempt_dir.glob("*.json"))),
            "path_sha256_manifest_sha256": raw_manifest_digest,
        },
        "decision": _decision(coverage, accounting),
    }


def _decision(coverage: dict[str, Any], accounting: dict[str, Any]) -> str:
    if any(accounting[arm]["cpu_hours"] <= 0 for arm in ("control", "treatment")):
        return "inconclusive"
    if any(
        coverage[scheme]["control"]["positions_per_cpu_hour"] <= 0
        for scheme in coverage
    ):
        return "inconclusive"
    if any(
        coverage[scheme]["treatment_control_yield_ratio"] is None
        or coverage[scheme]["treatment_control_yield_ratio"] < 1.15
        for scheme in coverage
    ):
        return "do_not_propose_larger_generation_run"
    return "propose_larger_generation_run"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--expected-freeze-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.root.resolve(), args.expected_freeze_sha256),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
