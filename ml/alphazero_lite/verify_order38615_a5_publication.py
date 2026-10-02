"""Independently verify public A5 result hashes and derived decision fields.

The original publisher remains the authority for validating the frozen runtime,
suites, and raw arena reports. This verifier adds checks over the published
results, matrix, and game ledger against #394's immutable amendment anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import publish_order38615_a5_diagnostic as publisher

ROOT = publisher.ROOT
DATA = publisher.DATA
AMENDMENT = publisher.AMENDMENT


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha_bytes(path.read_bytes())
    if actual != expected:
        raise ValueError(f"publication_{label}_hash_mismatch")


def validate_public_artifact_hashes(
    anchors: dict[str, Any],
    results_bytes: bytes,
    matrix_bytes: bytes,
    accounting_bytes: bytes,
) -> None:
    """Check amendment-anchored public artifact bytes, regardless of child refs."""
    for label, payload, key in (
        ("results", results_bytes, "results_sha256"),
        ("matrix", matrix_bytes, "opening_score_matrix_sha256"),
        ("accounting", accounting_bytes, "game_outcome_accounting_sha256"),
    ):
        if sha_bytes(payload) != anchors[key]:
            raise ValueError(f"publication_{label}_hash_mismatch")


def validate_public_bindings(
    amendment: dict[str, Any], results: dict[str, Any], matrix: dict[str, Any]
) -> None:
    """Enforce amendment anchors even when child references are resealed."""
    anchors = amendment["original_hashes"]
    results_binding = anchors["public_results_binding"]
    if results_binding != {
        "registration_sha256": anchors["registration_sha256"],
        "exclusion_manifest_sha256": anchors["manifest_sha256"],
        "evaluation_binding_sha256": anchors["evaluation_binding_sha256"],
    }:
        raise ValueError("publication_amendment_results_binding_mismatch")
    child_refs = {
        "registration_sha256": anchors["registration_sha256"],
        "evaluation_binding_sha256": anchors["evaluation_binding_sha256"],
        "exclusion_manifest_sha256": anchors["manifest_sha256"],
        "opening_score_matrix_sha256": anchors["opening_score_matrix_sha256"],
        "game_outcome_accounting_sha256": anchors["game_outcome_accounting_sha256"],
    }
    for key, expected in child_refs.items():
        if results.get(key) != expected:
            raise ValueError(f"publication_results_{key}_mismatch")
        matrix_key = {
            "exclusion_manifest_sha256": "exclusion_manifest_sha256",
            "game_outcome_accounting_sha256": "game_outcome_accounting_sha256",
        }.get(key)
        if matrix_key and matrix.get(matrix_key) != expected:
            raise ValueError(f"publication_matrix_{matrix_key}_mismatch")


def _ledger_summary(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        seed = str(row["seed"])
        grouped.setdefault(seed, {}).setdefault(int(row["opening_index"]), []).append(
            row
        )
    scores: dict[str, list[float]] = {}
    outcomes: dict[str, Any] = {}
    for seed, openings in grouped.items():
        opening_scores = []
        wins = draws = losses = 0
        seats: dict[str, dict[str, int | float]] = {}
        for seat in (0, 1):
            seat_rows = [
                row
                for rows_for_opening in openings.values()
                for row in rows_for_opening
                if int(row["challenger_player"]) == seat
            ]
            seat_wins = sum(row["winner"] == "challenger" for row in seat_rows)
            seat_draws = sum(row["winner"] == "draw" for row in seat_rows)
            seat_losses = sum(row["winner"] == "current" for row in seat_rows)
            seats[str(seat)] = {
                "games": len(seat_rows),
                "wins": seat_wins,
                "draws": seat_draws,
                "losses": seat_losses,
                "score": (seat_wins + 0.5 * seat_draws) / len(seat_rows),
            }
        for opening_index in sorted(openings):
            games = openings[opening_index]
            if len(games) != 2 or {
                int(game["challenger_player"]) for game in games
            } != {0, 1}:
                raise ValueError(
                    f"publication_accounting_opening_pair_invalid:{seed}:{opening_index}"
                )
            opening_wins = sum(game["winner"] == "challenger" for game in games)
            opening_draws = sum(game["winner"] == "draw" for game in games)
            opening_losses = sum(game["winner"] == "current" for game in games)
            wins += opening_wins
            draws += opening_draws
            losses += opening_losses
            opening_scores.append((opening_wins + 0.5 * opening_draws) / 2)
        scores[seed] = opening_scores
        outcomes[seed] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "games": wins + draws + losses,
            "score": (wins + 0.5 * draws) / (wins + draws + losses),
            "seat_outcomes": seats,
        }
    return scores, outcomes


def _decision(
    scores: dict[str, list[float]], registration: dict[str, Any]
) -> dict[str, Any]:
    analysis = registration["analysis"]
    if set(scores) != set(analysis["per_suite_seeds"]):
        raise ValueError("publication_matrix_suite_set_mismatch")
    intervals: dict[str, list[float]] = {}
    means: dict[str, float] = {}
    for suite, values in scores.items():
        data = np.asarray(values, dtype=np.float64)
        expected_n = int(registration["evaluation"]["suites"][suite]["opening_count"])
        if data.shape != (expected_n,) or not np.isfinite(data).all():
            raise ValueError(f"publication_matrix_opening_scores_invalid:{suite}")
        rng = np.random.default_rng(int(analysis["per_suite_seeds"][suite]))
        draws = data[
            rng.integers(
                0, len(data), size=(int(analysis["per_suite_resamples"]), len(data))
            )
        ].mean(axis=1)
        means[suite] = float(data.mean())
        intervals[suite] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    suite_pass = all(
        means[suite] >= analysis["success_rule"]["each_score_at_least"]
        and intervals[suite][0]
        > analysis["success_rule"]["each_interval_lower_strictly_above"]
        for suite in scores
    )
    seeds = sorted(scores, key=int)
    rng = np.random.default_rng(int(analysis["pooled_seed"]))
    count = int(analysis["pooled_resamples"])
    picks = [
        rng.integers(0, len(scores[seed]), size=(count, len(scores[seed])))
        for seed in seeds
    ]
    pooled_draws = np.mean(
        [
            np.asarray(scores[seed])[pick].mean(axis=1)
            for seed, pick in zip(seeds, picks)
        ],
        axis=0,
    )
    pooled_mean = float(np.mean([means[seed] for seed in seeds]))
    pooled_interval = [float(x) for x in np.quantile(pooled_draws, [0.025, 0.975])]
    pooled_pass = (
        pooled_mean >= analysis["success_rule"]["pooled_score_at_least"]
        and pooled_interval[0]
        > analysis["success_rule"]["pooled_interval_lower_strictly_above"]
    )
    return {
        "suite_pass": suite_pass,
        "pooled_pass": pooled_pass,
        "per_suite_and_pooled_pass": suite_pass and pooled_pass,
        "scores": means,
        "intervals": intervals,
        "pooled_score": pooled_mean,
        "pooled_interval": pooled_interval,
    }


def verify() -> dict[str, Any]:
    # Performs strict historical raw-evidence, runtime, suite, and exclusion checks.
    publisher.validate_publication()
    amendment = _read_json(AMENDMENT)
    results_path = DATA / "results.json"
    matrix_path = DATA / "opening-score-matrix.json"
    accounting_path = DATA / "game-outcome-accounting.jsonl"
    anchors = amendment["original_hashes"]
    validate_public_artifact_hashes(
        anchors,
        results_path.read_bytes(),
        matrix_path.read_bytes(),
        accounting_path.read_bytes(),
    )
    results, matrix = _read_json(results_path), _read_json(matrix_path)
    validate_public_bindings(amendment, results, matrix)
    rows = [
        json.loads(line)
        for line in accounting_path.read_text().splitlines()
        if line.strip()
    ]
    # Reconstruct the published ledger directly from the already-validated raw records.
    binding = _read_json(publisher.BIND)
    expected_rows = []
    for seed, record in binding["reports"].items():
        raw_rows = [
            json.loads(line)
            for line in Path(record["games"]).read_text().splitlines()
            if line.strip()
        ]
        for row in raw_rows:
            expected_rows.append(
                {
                    "seed": int(seed),
                    "game_index": int(row["game_index"]),
                    "opening_index": int(row["opening_index"]),
                    "game_within_opening": int(row["game_within_opening"]),
                    "challenger_player": int(row["challenger_player"]),
                    "winner": str(row["winner"]),
                    "margin": int(row["margin"]),
                    "game_length": int(row["game_length"]),
                    "opening_state_hash": str(row["opening_state_hash"]),
                }
            )
    expected_rows.sort(key=lambda row: (row["seed"], row["game_index"]))
    if rows != expected_rows:
        raise ValueError("publication_accounting_raw_outcome_mismatch")
    scores, outcomes = _ledger_summary(rows)
    if matrix["opening_scores"] != scores:
        raise ValueError("publication_matrix_outcomes_mismatch")
    if results["outcomes"] != outcomes:
        raise ValueError("publication_results_outcomes_mismatch")
    decision = _decision(scores, _read_json(publisher.REG))
    if any(
        results.get(key) != decision[key]
        for key in ("suite_pass", "pooled_pass", "per_suite_and_pooled_pass")
    ):
        raise ValueError("publication_decision_flag_mismatch")
    if (
        results["pooled_score"] != decision["pooled_score"]
        or results["pooled_interval_95_percentile"] != decision["pooled_interval"]
    ):
        raise ValueError("publication_decision_summary_mismatch")
    for seed, expected_score in decision["scores"].items():
        published = results["suite_results"][seed]
        if (
            published["score"] != expected_score
            or published["interval_95_percentile"] != decision["intervals"][seed]
            or published["wins"] != outcomes[seed]["wins"]
            or published["draws"] != outcomes[seed]["draws"]
            or published["losses"] != outcomes[seed]["losses"]
        ):
            raise ValueError(f"publication_suite_decision_mismatch:{seed}")
    return {
        "status": "verified_publication_and_recomputed_decision",
        "games": len(rows),
        "decision": decision["per_suite_and_pooled_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
