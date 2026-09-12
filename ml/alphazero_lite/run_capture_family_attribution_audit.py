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
import random
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
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT

SHADOW = ROOT / "docs/data/alphazero-lite-uniform1200-exact-shadow-replay.json"
EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
NEIGHBORHOODS = (
    ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json"
)
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
ROWMATCHED_ROOT = Path("/home/alex/Mancala/rowmatched-work/runs")
PR290_ROOT = ROOT / ".tmp/midgame-2x2-training/runs"
PR290_LANES = {
    "B": "control_like_exposure__unsharpened",
    "C": "uniform_exposure__unsharpened",
    "D": "control_like_exposure__sharpened",
}
EXPECTED_ROWMATCHED_WEIGHTS_JSON = {
    44: "7aea6ef5e0bcb3caede9a7e88ac9080cac12110d25770c65d605701770fcdd84",
    45: "c27f24f678b60c768f722d186b286cb197ae111bc75e5342628ffd21f53d5a21",
    46: "3c80a84081ceb43f981056caa528eefa96b914d215be41c453cef263300102de",
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


def _rowmatched_dir(seed: int) -> Path:
    return (
        ROWMATCHED_ROOT
        / f"seed{seed}"
        / "uniform1200_rowmatched"
        / f"uniform1200-rowmatched-seed{seed}-iter1"
    )


def _pr290_dir(seed: int, cell: str) -> Path:
    lane = PR290_LANES[cell]
    return PR290_ROOT / f"seed{seed}" / lane / f"midgame-2x2-seed{seed}-{lane}-iter1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _decoded_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for row in _rows(path):
        state = (
            row["state"]
            if isinstance(row["state"], dict)
            else decode_kalah_v3_base_state(list(row["state"]))
        )
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


def _searched_checkpoint(
    exact: dict[str, Any], evaluator: CheckpointEvaluator
) -> dict[str, Any]:
    """Parent checkpoint has no exported artifact; run the same frozen PUCT directly."""
    game = KalahGame.from_state(exact["state"])
    search = PUCT(
        evaluator,
        simulations=384,
        c_puct=1.25,
        rng=random.Random(42),
        **build_eval_search_options(),
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    selected = search.select_root_move(root, game.possible_moves())
    children = [
        {
            "move": move,
            "visits": root.children[move].visit_count,
            "q_value": root.children[move].q_value,
        }
        for move in game.possible_moves()
    ]
    prior, _ = evaluator.evaluate(game)
    return {
        "selected_action": selected,
        "regret": exact_regret(exact, selected),
        "visits": [float(value) for value in visits],
        "root_prior": [float(value) for value in prior],
        "child_q": children,
        "search_value": search.root_summary().get("root_q_value"),
    }


def target_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Score stored, legal-masked training policies only where exact labels exist."""
    return [
        {
            "canonical_state": row["canonical_state"],
            **policy_quality(row["policy"], labels[row["canonical_state"]]),
        }
        for row in rows
        if row["canonical_state"] in labels
    ]


def summarize_quality(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "rows": 0,
            "top_optimal_rate": None,
            "optimal_mass": None,
            "expected_exact_regret": None,
            "entropy": None,
            "max_probability": None,
        }
    return {
        "rows": len(rows),
        "top_optimal_rate": sum(bool(row["top_optimal"]) for row in rows) / len(rows),
        "optimal_mass": sum(float(row["optimal_mass"]) for row in rows) / len(rows),
        "expected_exact_regret": sum(
            float(row["expected_exact_regret"]) for row in rows
        )
        / len(rows),
        "entropy": sum(float(row["entropy"]) for row in rows) / len(rows),
        "max_probability": sum(float(row["max_probability"]) for row in rows)
        / len(rows),
    }


def neighborhood_coverage(
    rows: list[dict[str, Any]], cohort: list[dict[str, Any]], anchors: set[str]
) -> dict[str, dict[str, float | int]]:
    """Use only existing PR #291--#296 exact states; never synthesize neighbors."""
    keys = {row["canonical_state"] for row in rows}
    result: dict[str, dict[str, float | int]] = {}
    for radius in (0, 1, 2):
        states = {
            row["canonical_state_key"]
            for row in cohort
            if row.get("oracle_status") == "exact_solved"
            and row.get("radius", 0) <= radius
            and any(item["anchor_id"] in anchors for item in row.get("provenance", []))
        }
        occurrences = sum(row["canonical_state"] in states for row in rows)
        present = len(states & keys)
        result[str(radius)] = {
            "exact_states": len(states),
            "unique_state_coverage": present,
            "row_weighted_occurrences": occurrences,
            "coverage_fraction": present / len(states) if states else 0.0,
        }
    return result


def teacher_student_inversions(
    control_targets: list[dict[str, Any]],
    uniform_targets: list[dict[str, Any]],
    control_evaluations: list[dict[str, Any]],
    uniform_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare mean shared teacher regret with the matched raw student regret."""

    def mean_by_key(rows: list[dict[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(row["canonical_state"], []).append(
                float(row["expected_exact_regret"])
            )
        return {key: sum(values) / len(values) for key, values in grouped.items()}

    controls, uniforms = mean_by_key(control_targets), mean_by_key(uniform_targets)
    raw_control = {
        row["id"]: float(row["raw"]["regret"]) for row in control_evaluations
    }
    raw_uniform = {
        row["id"]: float(row["raw"]["regret"]) for row in uniform_evaluations
    }
    by_key = {row["canonical_state"]: row["id"] for row in capture_rows()}
    shared = sorted(set(controls) & set(uniforms) & set(by_key))
    inversions = [
        by_key[key]
        for key in shared
        if uniforms[key] <= controls[key]
        and raw_uniform[by_key[key]] > raw_control[by_key[key]]
    ]
    return {
        "shared_states": len(shared),
        "inversion_ids": inversions,
        "inversion_rate": len(inversions) / len(shared) if shared else None,
    }


def paired_target_deltas(
    control_targets: list[dict[str, Any]], uniform_targets: list[dict[str, Any]]
) -> dict[str, Any]:
    """Unique-state paired target deltas, uniform minus matched control."""

    def by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["canonical_state"], []).append(row)
        return {
            key: {
                metric: sum(float(row[metric]) for row in values) / len(values)
                for metric in ("optimal_mass", "expected_exact_regret", "entropy")
            }
            | {
                "top_optimal": sum(bool(row["top_optimal"]) for row in values)
                / len(values)
            }
            for key, values in grouped.items()
        }

    control, uniform = by_key(control_targets), by_key(uniform_targets)
    shared = sorted(set(control) & set(uniform))
    if not shared:
        return {
            "shared_states": 0,
            "optimal_mass_delta": None,
            "expected_exact_regret_delta": None,
            "entropy_delta": None,
            "top_action_correctness_delta": None,
        }
    return {
        "shared_states": len(shared),
        "optimal_mass_delta": sum(
            uniform[key]["optimal_mass"] - control[key]["optimal_mass"]
            for key in shared
        )
        / len(shared),
        "expected_exact_regret_delta": sum(
            uniform[key]["expected_exact_regret"]
            - control[key]["expected_exact_regret"]
            for key in shared
        )
        / len(shared),
        "entropy_delta": sum(
            uniform[key]["entropy"] - control[key]["entropy"] for key in shared
        )
        / len(shared),
        "top_action_correctness_delta": sum(
            uniform[key]["top_optimal"] - control[key]["top_optimal"] for key in shared
        )
        / len(shared),
    }


def comparative_search_regression(
    control_raw: float, control_search: float, uniform_raw: float, uniform_search: float
) -> bool:
    """A matched raw tie/better uniform is separated only by production search."""
    return uniform_raw <= control_raw and uniform_search > control_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    positions = capture_rows()
    shadow = json.loads(SHADOW.read_text())
    critical = decision_critical_ids(shadow)
    cohort = json.loads(NEIGHBORHOODS.read_text())["cohort"]
    labels = {row["canonical_state"]: row for row in positions}
    inputs: dict[str, Any] = {
        "exact_reference": {"path": str(EXACT), "sha256": sha256_file(EXACT)},
        "shadow": {"path": str(SHADOW), "sha256": sha256_file(SHADOW)},
        "neighborhoods": {
            "path": str(NEIGHBORHOODS),
            "sha256": sha256_file(NEIGHBORHOODS),
        },
        "checkpoints": {},
        "replays": {},
        "fixed_replays": {},
    }
    evaluations: dict[str, Any] = {}
    parent_evaluations: dict[str, Any] = {}
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
        parent = _run_dir(seed, "control_384_192") / "parent_init_checkpoint.npz"
        parent_sha = sha256_file(parent)
        inputs["checkpoints"][f"{seed}:parent"] = {
            "path": str(parent),
            "sha256": parent_sha,
        }
        parent_evaluator = CheckpointEvaluator(parent, input_encoding="kalah_v3")
        parent_evaluations[str(seed)] = [
            {
                "id": row["id"],
                "raw": _raw(row, parent_evaluator),
                "search": _searched_checkpoint(row, parent_evaluator),
            }
            for row in positions
        ]
    fixed_rows = []
    for path in FIXED:
        inputs["fixed_replays"][path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        fixed_rows.extend(_decoded_rows(path))
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
    target_quality = {
        "fixed": summarize_quality(target_rows(fixed_rows, labels)),
        "dynamic": {
            f"{seed}:{lane}": summarize_quality(
                target_rows(replay_by_lane[seed, lane], labels)
            )
            for seed in (44, 45, 46)
            for lane in ("control_384_192", "uniform1200")
        },
    }
    inversions = {
        str(seed): teacher_student_inversions(
            target_rows(replay_by_lane[seed, "control_384_192"], labels),
            target_rows(replay_by_lane[seed, "uniform1200"], labels),
            evaluations[f"{seed}:control_384_192"],
            evaluations[f"{seed}:uniform1200"],
        )
        for seed in (44, 45, 46)
    }
    paired_targets = {
        str(seed): paired_target_deltas(
            target_rows(replay_by_lane[seed, "control_384_192"], labels),
            target_rows(replay_by_lane[seed, "uniform1200"], labels),
        )
        for seed in (44, 45, 46)
    }
    critical_positions = [row for row in positions if row["id"] in critical]
    mechanisms: dict[str, list[dict[str, Any]]] = {}
    for seed in (44, 45, 46):
        control_targets = {
            row["canonical_state"]: row
            for row in target_rows(replay_by_lane[seed, "control_384_192"], labels)
        }
        uniform_targets = {
            row["canonical_state"]: row
            for row in target_rows(replay_by_lane[seed, "uniform1200"], labels)
        }
        control_eval = {
            row["id"]: row for row in evaluations[f"{seed}:control_384_192"]
        }
        uniform_eval = {row["id"]: row for row in evaluations[f"{seed}:uniform1200"]}
        mechanisms[str(seed)] = []
        for exact in critical_positions:
            identifier, key = exact["id"], exact["canonical_state"]
            control, uniform = control_eval[identifier], uniform_eval[identifier]
            c_raw, c_search = (
                float(control["raw"]["regret"]),
                float(control["search"]["regret"]),
            )
            u_raw, u_search = (
                float(uniform["raw"]["regret"]),
                float(uniform["search"]["regret"]),
            )
            has_pair = key in control_targets and key in uniform_targets
            coverage_adequate = (
                coverage[str(seed)]["uniform1200"][identifier]
                >= coverage[str(seed)]["control_384_192"][identifier]
            )
            target_worse = has_pair and float(
                uniform_targets[key]["expected_exact_regret"]
            ) > float(control_targets[key]["expected_exact_regret"])
            search_stage = comparative_search_regression(
                c_raw, c_search, u_raw, u_search
            )
            category = (
                mechanism(
                    search_induced=search_stage,
                    coverage_adequate=coverage_adequate,
                    target_worse=target_worse,
                    raw_worse=u_raw > c_raw,
                )
                if u_search > c_search
                else "mixed_or_unclear"
            )
            mechanisms[str(seed)].append(
                {
                    "id": identifier,
                    "category": category,
                    "control_raw_regret": c_raw,
                    "control_search_regret": c_search,
                    "uniform_raw_regret": u_raw,
                    "uniform_search_regret": u_search,
                    "coverage_adequate": coverage_adequate,
                    "paired_target_evidence": has_pair,
                    "comparative_search_regression": search_stage,
                }
            )
    dominant = cross_seed_dominance(
        {
            int(seed): [row["category"] for row in rows]
            for seed, rows in mechanisms.items()
        }
    )
    final_classification = (
        "capture_regression_search_induced"
        if dominant == "inference_search_degradation"
        else "capture_regression_mechanism_heterogeneous"
    )
    neighborhoods = {
        f"{seed}:{lane}": {
            "all_capture": neighborhood_coverage(
                replay_by_lane[seed, lane], cohort, {row["id"] for row in positions}
            ),
            "decision_critical": neighborhood_coverage(
                replay_by_lane[seed, lane], cohort, set(critical)
            ),
        }
        for seed in (44, 45, 46)
        for lane in ("control_384_192", "uniform1200")
    }
    secondary: dict[str, Any] = {"rowmatched": {}, "pr290": {}}
    for seed in (44, 45, 46):
        directory = _rowmatched_dir(seed)
        checkpoint = directory / "checkpoint.npz"
        actual = sha256_file(checkpoint)
        metadata = json.loads((directory / "metadata.json").read_text())
        artifacts = metadata.get("artifacts", {})
        if artifacts.get("weights_sha256") != actual:
            raise RuntimeError(
                f"rowmatched checkpoint SHA disagrees with metadata for seed {seed}"
            )
        if (
            artifacts.get("weights_json_sha256")
            != EXPECTED_ROWMATCHED_WEIGHTS_JSON[seed]
        ):
            raise RuntimeError(f"rowmatched weights SHA mismatch for seed {seed}")
        evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
        secondary["rowmatched"][str(seed)] = {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": actual,
                "recorded_weights_json_sha256": artifacts["weights_json_sha256"],
            },
            "rows": [
                {
                    "id": row["id"],
                    "raw": _raw(row, evaluator),
                    "search": _searched(row, directory),
                }
                for row in critical_positions
            ],
        }
        secondary["pr290"][str(seed)] = {}
        for cell in ("B", "C", "D"):
            directory = _pr290_dir(seed, cell)
            checkpoint = directory / "checkpoint.npz"
            actual = sha256_file(checkpoint)
            metadata = json.loads((directory / "metadata.json").read_text())
            artifacts = metadata.get("artifacts", {})
            expected = shadow["pr290"][cell][str(seed)]["checkpoint_sha256"]
            if artifacts.get("weights_sha256") != actual:
                raise RuntimeError(
                    f"PR #290 {cell} checkpoint metadata mismatch for seed {seed}"
                )
            if artifacts.get("weights_json_sha256") != expected:
                raise RuntimeError(
                    f"PR #290 {cell} historical SHA mismatch for seed {seed}"
                )
            evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
            secondary["pr290"][str(seed)][cell] = {
                "available": True,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": actual,
                    "recorded_weights_json_sha256": expected,
                },
                "rows": [
                    {
                        "id": row["id"],
                        "raw": _raw(row, evaluator),
                        "search": _searched(row, directory),
                    }
                    for row in critical_positions
                ],
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
        "parent_evaluations": parent_evaluations,
        "root_replay_coverage": coverage,
        "stored_target_quality": target_quality,
        "teacher_student_inversions": inversions,
        "paired_target_deltas": paired_targets,
        "neighborhood_coverage": neighborhoods,
        "secondary": secondary,
        "mechanisms": mechanisms,
        "cross_seed_dominant_mechanism": dominant,
        "hard_classification": final_classification,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
