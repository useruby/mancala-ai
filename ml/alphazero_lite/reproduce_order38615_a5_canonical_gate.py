"""Publish and reproduce #392's canonical prefilter opening score matrix."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "docs/data/order38615-a5-promotion-review-result.json"
PLAN = ROOT / "docs/data/order38615-a5-promotion-review-plan.json"
RUN = ROOT / ".tmp/order38615-a5-promotion-review"
SUITE = (
    ROOT / "docs/data/alphazero-lite-production-prefilter-calibration-openings-v1.jsonl"
)
OUT = ROOT / "docs/data/order38615-a5-canonical-gate/opening-score-matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce() -> dict[str, Any]:
    recorded = json.loads(RESULT.read_text())["gate"]["canonical_prefilter"]
    gate_result = json.loads(RESULT.read_text())["gate"]
    report_path = RUN / "candidate_vs_current_arena.json"
    games_path = RUN / "shadow_prefilter_games.jsonl"
    gate_report_path = RUN / "gate_report.json"
    ledgers = {
        "seed_ledger_sha256": RUN / "shadow_prefilter_seed_ledger.jsonl",
        "search_configuration_ledger_sha256": RUN
        / "shadow_prefilter_search_configuration_ledger.jsonl",
        "search_outcome_ledger_sha256": RUN
        / "shadow_prefilter_search_outcome_ledger.jsonl",
    }
    if sha(gate_report_path) != gate_result["report_sha256"]:
        raise ValueError("original_gate_report_hash_mismatch")
    report = json.loads(report_path.read_text())
    if sha(report_path) != recorded["arena_report_sha256"]:
        raise ValueError("original_canonical_report_hash_mismatch")
    if sha(games_path) != recorded["game_rows_sha256"]:
        raise ValueError("original_canonical_game_rows_hash_mismatch")
    for key, path in ledgers.items():
        if sha(path) != recorded[key]:
            raise ValueError(f"original_canonical_ledger_hash_mismatch:{key}")
    suite_sha = sha(SUITE)
    if suite_sha != recorded["suite_sha256"]:
        raise ValueError("canonical_suite_hash_mismatch")
    games = [
        json.loads(line) for line in games_path.read_text().splitlines() if line.strip()
    ]
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    if len(games) != 512 or len(openings) != 256 or report["games_played"] != 512:
        raise ValueError("canonical_gate_accounting_mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        grouped.setdefault(int(game["opening_index"]), []).append(game)
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, opening in enumerate(openings):
        pair = sorted(
            grouped.get(index, []), key=lambda row: row["game_within_opening"]
        )
        if len(pair) != 2 or {row["challenger_player"] for row in pair} != {0, 1}:
            raise ValueError(f"canonical_pairing_mismatch:{index}")
        if any(row["opening_prefix_moves"] != opening["prefix_moves"] for row in pair):
            raise ValueError(f"canonical_opening_mapping_mismatch:{index}")
        pair_scores = [
            0.5
            if row["winner"] == "draw"
            else (1.0 if row["winner"] == "challenger" else 0.0)
            for row in pair
        ]
        score = float(sum(pair_scores) / 2)
        scores.append(score)
        rows.append(
            {
                "opening_index": index,
                "canonical_resulting_state_hash": opening[
                    "canonical_resulting_state_hash"
                ],
                "score": score,
                "game_scores": pair_scores,
                "challenger_seats": [int(row["challenger_player"]) for row in pair],
            }
        )
    pair_source = json.loads((RUN / "shadow_prefilter_opening_pairs.json").read_text())
    scores_array = np.asarray(scores, dtype=np.float64)
    if not np.array_equal(scores_array, np.asarray(pair_source["pair_scores"])):
        raise ValueError("canonical_pair_scores_do_not_match_original_gate")
    rng = random.Random(331)
    bootstrap = sorted(
        sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(20000)
    )
    interval = [float(bootstrap[500]), float(bootstrap[19499])]
    if interval != [
        pair_source["pair_bootstrap_ci95"]["lower"],
        pair_source["pair_bootstrap_ci95"]["upper"],
    ]:
        raise ValueError("canonical_bootstrap_reproduction_mismatch")
    matrix = {
        "schema": "order38615-a5-canonical-prefilter-opening-score-matrix-v1",
        "plan_sha256": sha(PLAN),
        "gate_result_sha256": sha(RESULT),
        "canonical_suite_sha256": suite_sha,
        "original_arena_report_sha256": sha(report_path),
        "original_game_rows_sha256": sha(games_path),
        "seed_ledger_sha256": recorded["seed_ledger_sha256"],
        "search_configuration_ledger_sha256": recorded[
            "search_configuration_ledger_sha256"
        ],
        "search_outcome_ledger_sha256": recorded["search_outcome_ledger_sha256"],
        "opening_count": len(rows),
        "games": len(games),
        "wins": report["wins"],
        "draws": report["draws"],
        "losses": report["losses"],
        "score": float(scores_array.mean()),
        "bootstrap": {
            "method": "opening-pair percentile bootstrap",
            "seed": 331,
            "resamples": 20000,
            "order_statistics_zero_based": [500, 19499],
            "interval_95_percentile": interval,
        },
        "opening_scores": rows,
        "reproduction": {
            "score_matches_recorded": float(scores_array.mean())
            == recorded["seat_paired_score"],
            "interval_matches_recorded": interval
            == recorded["opening_pair_interval_95"],
            "decision": "fail: canonical score below 0.55; no promotion",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in matrix.items() if key != "opening_scores"},
            indent=2,
        )
    )
    print(f"matrix_path={OUT.relative_to(ROOT)}")
    print(f"matrix_sha256={sha(OUT)}")
    return matrix


if __name__ == "__main__":
    reproduce()
