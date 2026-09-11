#!/usr/bin/env python3
# ruff: noqa: E402
"""Read-only causal attribution audit for the exact capture_available family.

This module deliberately has no training, self-play, promotion, or oracle
generation imports.  It consumes frozen PR #287/#297 artifacts and fails
closed when a required historical input is unavailable or hash-mismatched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    build_eval_search_options,
    evaluate_artifact_position,
)
from ml.alphazero_lite.forensic_exact_references import exact_regret
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import decode_kalah_v3_base_state
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import CheckpointEvaluator

SHADOW = ROOT / "docs/data/alphazero-lite-uniform1200-exact-shadow-replay.json"
EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
RUNS = Path("/tmp/uniform1200-seed-confirmation/runs")
FIXED = (
    Path(
        "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl"
    ),
    Path("/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"),
)
EXPECTED_CHECKPOINTS = {
    44: {
        "control_384_192": "979f906a80610e2a6648d5bc2ac1a88d47efad75bda53488b045688d12b2c513",
        "uniform1200": "806d04b9452b98dbd7a844e553ea8e4e8af6094cbe37790cf16bc0a0bc90278a",
    },
    45: {
        "control_384_192": "52d3df027c0cdccaba7e27f1c2827f673091a610f0103271c77411f0dca4131d",
        "uniform1200": "97a82737efebcee541e9572def4cca08da5e0f5c12728f3a439753ed1b6594da",
    },
    46: {
        "control_384_192": "ed61f4a6e943529a5d4fe5d452b00380bad0927c423f983de8fa9a363adc4ee2",
        "uniform1200": "9baa2227602df7ffa0c16c009d16973c67181a079fca810a364b8f4ce97aa604",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rows() -> list[dict[str, Any]]:
    rows = json.loads(EXACT.read_text())["rows"]
    result = [row for row in rows if row["id"].startswith("capture_available-")]
    if len(result) != 24 or any(
        row["exact_status"] != "exact_solved" for row in result
    ):
        raise RuntimeError(
            "expected exactly 24 exact-solved capture_available positions"
        )
    return sorted(result, key=lambda row: row["id"])


def decision_critical_ids(shadow: dict[str, Any]) -> list[str]:
    """Mechanically select repeated exact regressions and failed-bucket causes."""
    counts: Counter[str] = Counter()
    for seed in ("44", "45", "46"):
        for row in shadow["decision_flips"][seed]["genuine_regressions"]:
            if (
                row["bucket"] == "capture_available"
                and row["uniform_minus_control_exact_regret"] > 0
            ):
                counts[row["id"]] += 1
    # A repeated regression necessarily explains each failing 24-row bucket.
    return sorted(identifier for identifier, count in counts.items() if count >= 2)


def policy_quality(
    policy: Iterable[float], exact: dict[str, Any]
) -> dict[str, float | bool | int]:
    values = {
        int(key): float(value) for key, value in exact["exact_action_values"].items()
    }
    legal = sorted(values)
    probabilities = np.asarray(list(policy), dtype=float)
    masked = np.zeros(6, dtype=float)
    masked[legal] = probabilities[legal]
    masked /= masked.sum() if masked.sum() else 1.0
    top = int(max(legal, key=lambda action: (masked[action], -action)))
    player = int(exact["state"]["current_player"])
    optimum = max(values.values()) if player == 0 else min(values.values())
    regrets = {
        action: (optimum - value if player == 0 else value - optimum)
        for action, value in values.items()
    }
    return {
        "top_action": top,
        "top_optimal": top in exact["exact_optimal_actions"],
        "optimal_mass": float(
            sum(masked[action] for action in exact["exact_optimal_actions"])
        ),
        "expected_exact_action_value": float(
            sum(masked[action] * values[action] for action in legal)
        ),
        "expected_exact_regret": float(
            sum(masked[action] * regrets[action] for action in legal)
        ),
        "entropy": float(
            -sum(value * math.log(value) for value in masked[legal] if value > 0)
        ),
        "max_probability": float(max(masked[legal])),
    }


def transition(raw_regret: float | None, search_regret: float | None) -> str:
    raw_ok, search_ok = raw_regret == 0, search_regret == 0
    return f"raw_{'correct' if raw_ok else 'wrong'}_to_search_{'correct' if search_ok else 'wrong'}"


def search_induced_regression(
    uniform_raw: float, control_raw: float, uniform_search: float
) -> bool:
    return (
        uniform_raw == 0 or uniform_raw <= control_raw
    ) and uniform_search > uniform_raw


def mechanism(
    *,
    search_induced: bool,
    coverage_adequate: bool,
    target_worse: bool,
    raw_worse: bool,
) -> str:
    if search_induced:
        return "inference_search_degradation"
    if coverage_adequate and target_worse:
        return "target_quality_deficit"
    if not coverage_adequate and not target_worse:
        return "coverage_deficit"
    if coverage_adequate and not target_worse and raw_worse:
        return "student_retention_failure"
    return "mixed_or_unclear"


def cross_seed_dominance(categories: dict[int, list[str]]) -> str | None:
    counts = Counter(
        category for values in categories.values() for category in set(values)
    )
    eligible = [(count, name) for name, count in counts.items() if count >= 2]
    return max(eligible)[1] if eligible else None


def _run_dir(seed: int, lane: str) -> Path:
    name = f"uniform1200-confirm-seed{seed}-{lane}-iter1"
    return RUNS / f"seed{seed}" / lane / name


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _decoded_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for row in _rows(path):
        state = decode_kalah_v3_base_state(list(row["state"]))
        result.append({**row, "canonical_state": canonical_state_key(state)})
    return result


def _raw(exact: dict[str, Any], evaluator: CheckpointEvaluator) -> dict[str, Any]:
    game = KalahGame.from_state(exact["state"])
    policy, value = evaluator.evaluate(game)
    quality = policy_quality(policy, exact)
    quality.update(
        {
            "policy": [float(item) for item in policy],
            "value": float(value),
            "regret": exact_regret(exact, quality["top_action"]),
        }
    )
    return quality


def _searched(exact: dict[str, Any], artifact: Path) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=artifact,
        state=exact["state"],
        simulations=384,
        seed=42,
        c_puct=1.25,
        search_options=build_eval_search_options(),
    )
    selected = result["selected_move"]
    return {
        "selected_action": selected,
        "regret": exact_regret(exact, selected),
        "visits": result["visits"],
        "root_prior": result["policy"],
        "child_q": result["child_stats"],
        "search_value": result.get("search_root_value", result.get("value")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    positions = capture_rows()
    shadow = json.loads(SHADOW.read_text())
    critical = decision_critical_ids(shadow)
    inputs: dict[str, Any] = {
        "exact_reference": {"path": str(EXACT), "sha256": sha256_file(EXACT)},
        "shadow": {"path": str(SHADOW), "sha256": sha256_file(SHADOW)},
        "checkpoints": {},
        "replays": {},
    }
    evaluations: dict[str, Any] = {}
    replay_by_lane: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in (44, 45, 46):
        for lane in ("control_384_192", "uniform1200"):
            directory = _run_dir(seed, lane)
            checkpoint, replay = (
                directory / "checkpoint.npz",
                directory / "self_play.jsonl",
            )
            actual = sha256_file(checkpoint)
            if actual != EXPECTED_CHECKPOINTS[seed][lane]:
                raise RuntimeError(f"checkpoint SHA mismatch for seed {seed} {lane}")
            inputs["checkpoints"][f"{seed}:{lane}"] = {
                "path": str(checkpoint),
                "sha256": actual,
            }
            inputs["replays"][f"{seed}:{lane}"] = {
                "path": str(replay),
                "sha256": sha256_file(replay),
            }
            decoded = _decoded_rows(replay)
            replay_by_lane[seed, lane] = decoded
            evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
            evaluations[f"{seed}:{lane}"] = [
                {
                    "id": row["id"],
                    "raw": _raw(row, evaluator),
                    "search": _searched(row, directory),
                }
                for row in positions
            ]
    coverage = {}
    exact_by_key = {row["canonical_state"]: row for row in positions}
    for seed in (44, 45, 46):
        coverage[str(seed)] = {}
        for lane in ("control_384_192", "uniform1200"):
            matching = [
                row
                for row in replay_by_lane[seed, lane]
                if row["canonical_state"] in exact_by_key
            ]
            coverage[str(seed)][lane] = {
                identifier: sum(
                    row["canonical_state"] == exact["canonical_state"]
                    for row in matching
                )
                for identifier, exact in ((row["id"], row) for row in positions)
            }
    result = {
        "schema": "azlite_capture_family_attribution_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "tablebase_generation": False,
        },
        "inputs": inputs,
        "capture_population": {
            "count": len(positions),
            "ids": [row["id"] for row in positions],
        },
        "decision_critical_capture_set": {
            "ids": critical,
            "sha256": hashlib.sha256("\n".join(critical).encode()).hexdigest(),
        },
        "evaluations": evaluations,
        "root_replay_coverage": coverage,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
