"""Refresh replay policy teachers without changing their trajectories or values."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.exact_root_decision import exact_root_decision
from ml.alphazero_lite.exact_root_policy_targets import (
    SELECTED_ONE_HOT,
    exact_root_actual_policy_target_mode,
    exact_root_policy_target,
)
from ml.alphazero_lite.export_artifact import sha256_file
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.self_play import PUCT, build_policy_target
from ml.alphazero_lite.train import (
    derive_legal_moves_from_encoded_state,
    validate_policy_target,
)

POLICY_FIELDS = {
    "policy",
    "stored_policy_target",
    "top_target_move",
    "policy_target_mode",
    "policy_target_actual_mode",
    "policy_target_noise_mode",
    "action_sampling_noise_enabled",
    "dirichlet_alpha",
    "dirichlet_epsilon_for_sampling",
    "dirichlet_epsilon_for_target",
    "target_dirichlet_epsilon",
    "sampling_dirichlet_epsilon",
    "simulations",
    "root_visit_counts",
    "teacher_source",
    "policy_teacher_provenance",
    "active_pit_stones",
    "exact_action_margins",
    "exact_optimal_actions",
    "exact_selected_action",
    "exact_root_tie_rule",
    "solver_implementation_identity",
    "native_probe_sha256",
    "tablebase_sha256",
    "puct_simulations_executed",
    "search_top1",
    "search_top1_visit_share",
    "raw_top1",
    "raw_margin",
    "kl_search_raw",
}


def game_from_encoded_state(state: list[float]) -> KalahGame:
    values = [int(round(float(value) * 48)) for value in state[:14]]
    current_player = int(round(float(state[14])))
    return KalahGame.from_state(
        {
            "player_pits": values[:6],
            "opponent_pits": values[6:12],
            "player_store": values[12],
            "opponent_store": values[13],
            "current_player": current_player,
        }
    )


def active_stones(row: dict[str, Any]) -> int:
    return int(sum(game_from_encoded_state(row["state"]).pits))


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"replay-policy-refresh-v1:{state_hash}".encode()).hexdigest()[
            :16
        ],
        16,
    )


def _entropy(policy: list[float]) -> float:
    return -sum(value * math.log(value) for value in policy if value > 0.0)


def _js(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    return 0.5 * sum(
        a * math.log(a / m) for a, m in zip(left, midpoint) if a
    ) + 0.5 * sum(b * math.log(b / m) for b, m in zip(right, midpoint) if b)


def _cross_entropy(source: list[float], target: list[float]) -> float | None:
    if any(
        probability > 0.0 and target[index] <= 0.0
        for index, probability in enumerate(source)
    ):
        return None
    return -sum(
        probability * math.log(target[index])
        for index, probability in enumerate(source)
        if probability
    )


def refresh_target(
    row: dict[str, Any],
    *,
    evaluator: ArtifactEvaluator,
    tablebase: NativeExactRootTablebase,
    runtime_policy: dict[str, Any],
    simulations: int = 1200,
) -> tuple[list[float], dict[str, Any], str, int]:
    """Apply the production exact-root handoff or denoised PUCT teacher."""
    game = game_from_encoded_state(row["state"])
    legal = game.possible_moves()
    decision = exact_root_decision(
        game,
        evaluator,
        tablebase=tablebase,
        threshold=int(runtime_policy["exact_root_threshold"]),
    )
    if decision is not None:
        policy = exact_root_policy_target(
            selected_action=decision.selected_move,
            optimal_actions=decision.optimal_moves,
            mode=SELECTED_ONE_HOT,
        )
        metadata = {
            "policy_target_mode": "sharpened",
            "policy_target_actual_mode": exact_root_actual_policy_target_mode(
                SELECTED_ONE_HOT
            ),
            "policy_target_noise_mode": "denoised",
            "action_sampling_noise_enabled": False,
            "dirichlet_alpha": 0.0,
            "dirichlet_epsilon_for_sampling": 0.0,
            "dirichlet_epsilon_for_target": 0.0,
            "target_dirichlet_epsilon": 0.0,
            "sampling_dirichlet_epsilon": 0.0,
            "simulations": 0,
            "legal_moves": legal,
            "top_target_move": decision.selected_move,
            "stored_policy_target": policy,
            "teacher_source": "exact_root_tablebase",
            "active_pit_stones": decision.active_pit_stones,
            "exact_action_margins": decision.action_margins,
            "exact_optimal_actions": decision.optimal_moves,
            "exact_selected_action": decision.selected_move,
            "exact_root_tie_rule": runtime_policy["exact_root_tie_rule"],
            "solver_implementation_identity": runtime_policy[
                "solver_implementation_identity"
            ],
            "native_probe_sha256": runtime_policy["native_probe"]["sha256"],
            "tablebase_sha256": runtime_policy["tablebase"]["sha256"],
            "puct_simulations_executed": 0,
        }
        return policy, metadata, "exact_root16", decision.solver_calls
    search = PUCT(evaluator, simulations, 1.25, random.Random(_seed(row["state_hash"])))
    visits, _root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    # Preserve the production target-temperature schedule: 1.1 through ply 11,
    # then 0.15. The replay's move index is immutable trajectory metadata.
    temperature = 1.1 if int(row.get("move_index", 0)) < 12 else 0.15
    policy = build_policy_target(
        visits, legal_moves=legal, temperature=temperature, mode="sharpened"
    )
    metadata = {
        "policy_target_mode": "sharpened",
        "policy_target_actual_mode": "sharpened",
        "policy_target_noise_mode": "denoised",
        "action_sampling_noise_enabled": False,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon_for_sampling": 0.0,
        "dirichlet_epsilon_for_target": 0.0,
        "target_dirichlet_epsilon": 0.0,
        "sampling_dirichlet_epsilon": 0.0,
        "simulations": simulations,
        "legal_moves": legal,
        "top_target_move": int(np.argmax(policy)),
        "stored_policy_target": policy,
        "root_visit_counts": [int(value) for value in visits],
        "teacher_source": "seed455_denoised_puct",
        "puct_simulations_executed": simulations,
    }
    return policy, metadata, "puct1200_denoised", 0


def refresh_replay_policy_targets(
    source: Path,
    output: Path,
    *,
    parent_artifact: Path,
    runtime_policy_path: Path,
    native_probe: Path,
    tablebase_path: Path,
    refresh_version: str = "replay_policy_refresh_v1",
) -> dict[str, Any]:
    """Write an immutable replay copy with only policy teacher fields replaced."""
    runtime_policy = json.loads(runtime_policy_path.read_text(encoding="utf-8"))
    if (
        sha256_file(runtime_policy_path)
        != "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d"
    ):
        raise ValueError("runtime_search_policy_sha_mismatch")
    metadata = json.loads(
        (parent_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        metadata["artifacts"]["weights_json_sha256"]
        != "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c"
    ):
        raise ValueError("parent_weights_sha_mismatch")
    evaluator = ArtifactEvaluator(parent_artifact)
    counters: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        NativeExactRootTablebase(
            native_probe, tablebase_path, warm_on_start=True
        ) as solver,
        source.open(encoding="utf-8") as reader,
        output.open("w", encoding="utf-8") as writer,
    ):
        for line_number, line in enumerate(reader, 1):
            row = json.loads(line)
            legal = derive_legal_moves_from_encoded_state(row["state"])
            if legal != row.get("legal_moves"):
                raise ValueError(f"invalid_replay_row:{line_number}")
            policy, policy_metadata, mode, solver_calls = refresh_target(
                row,
                evaluator=evaluator,
                tablebase=solver,
                runtime_policy=runtime_policy,
            )
            transformed = copy.deepcopy(row)
            for field in POLICY_FIELDS:
                transformed.pop(field, None)
            transformed.update(policy_metadata)
            transformed["policy"] = policy
            transformed["policy_teacher_provenance"] = {
                "refresh_version": refresh_version,
                "parent_weights_sha256": metadata["artifacts"]["weights_json_sha256"],
                "runtime_policy_sha256": sha256_file(runtime_policy_path),
                "search_profile": mode,
            }
            validate_policy_target(
                np.asarray(policy),
                state=transformed["state"],
                path=output,
                row_number=line_number,
                policy_target_mode="sharpened",
                declared_mode=transformed.get("policy_target_actual_mode"),
                row=transformed,
            )
            if {
                key: value for key, value in row.items() if key not in POLICY_FIELDS
            } != {
                key: value
                for key, value in transformed.items()
                if key not in POLICY_FIELDS
            }:
                raise ValueError("opening_disagreement_policy_refresh_integrity_failed")
            counters["source_rows"] += 1
            counters[
                "rows_changed" if row.get("policy") != policy else "rows_unchanged"
            ] += 1
            counters[f"{mode}_rows"] += 1
            counters["exact_root_solver_calls"] += solver_calls
            counters["puct_simulations"] += 1200 if mode.startswith("puct") else 0
            writer.write(json.dumps(transformed, separators=(",", ":")) + "\n")
    counters["output_rows"] = sum(1 for _ in output.open(encoding="utf-8"))
    if counters["source_rows"] != counters["output_rows"]:
        raise ValueError("opening_disagreement_policy_refresh_integrity_failed")
    return {
        **dict(counters),
        "source_sha256": sha256_file(source),
        "refreshed_sha256": sha256_file(output),
        "state_mismatches": 0,
        "value_mismatches": 0,
        "winner_mismatches": 0,
        "ordering_mismatches": 0,
        "bucket_family_mismatches": 0,
        "legal_move_violations": 0,
    }


def audit_replay(source: Path) -> dict[str, Any]:
    """Summarize replay validity and label distributions without rewriting it."""
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    seen: set[str] = set()
    report: Counter[str] = Counter()
    report["rows"] = len(rows)
    for index, row in enumerate(rows, 1):
        legal = derive_legal_moves_from_encoded_state(row["state"])
        if legal != row.get("legal_moves"):
            raise ValueError(f"invalid_replay_row:{index}")
        validate_policy_target(
            np.asarray(row["policy"]),
            state=row["state"],
            path=source,
            row_number=index,
            policy_target_mode="sharpened",
            declared_mode=row.get("policy_target_actual_mode"),
            row=row,
        )
        report[f"active_{active_stones(row)}"] += 1
        report[f"legal_{len(legal)}"] += 1
        report["le16" if active_stones(row) <= 16 else "gt16"] += 1
        report[
            f"policy_mode_{row.get('policy_target_actual_mode', row.get('policy_target_mode'))}"
        ] += 1
        report[f"value_mode_{row.get('value_target_mode')}"] += 1
        report[f"bucket_{row.get('bucket')}"] += 1
        report[f"teacher_{row.get('teacher_source')}"] += 1
        report["duplicate_rows"] += row["state_hash"] in seen
        seen.add(row["state_hash"])
    return {
        "sha256": sha256_file(source),
        **dict(report),
        "duplicate_state_rate": report["duplicate_rows"] / len(rows),
    }


def target_drift_audit(
    source: Path, refreshed: Path, *, parent_artifact: Path
) -> dict[str, Any]:
    """Describe old/current teachers and parent raw-policy agreement by stone bucket."""
    original = [json.loads(line) for line in source.open(encoding="utf-8")]
    current = [json.loads(line) for line in refreshed.open(encoding="utf-8")]
    if len(original) != len(current):
        raise ValueError("opening_disagreement_policy_refresh_integrity_failed")
    evaluator = ArtifactEvaluator(parent_artifact)
    buckets: dict[str, list[dict[str, float | bool | None]]] = {}
    for old, new in zip(original, current):
        if old["state"] != new["state"]:
            raise ValueError("opening_disagreement_policy_refresh_integrity_failed")
        raw, _value = evaluator.evaluate(game_from_encoded_state(old["state"]))
        old_policy, new_policy = old["policy"], new["policy"]
        bucket = (
            "le16"
            if active_stones(old) <= 16
            else f"gt16_{active_stones(old) // 4 * 4}s"
        )
        buckets.setdefault(bucket, []).append(
            {
                "top1_agreement": int(np.argmax(old_policy) == np.argmax(new_policy)),
                "l1": float(
                    np.abs(np.asarray(old_policy) - np.asarray(new_policy)).sum()
                ),
                "js": _js(old_policy, new_policy),
                "old_to_new_ce": _cross_entropy(old_policy, new_policy),
                "new_to_old_ce": _cross_entropy(new_policy, old_policy),
                "old_entropy": _entropy(old_policy),
                "current_entropy": _entropy(new_policy),
                "argmax_changed": bool(np.argmax(old_policy) != np.argmax(new_policy)),
                "substantial_mass_movement": bool(
                    np.abs(np.asarray(old_policy) - np.asarray(new_policy)).sum() >= 0.5
                ),
                "raw_old_top1": int(np.argmax(raw) == np.argmax(old_policy)),
                "raw_current_top1": int(np.argmax(raw) == np.argmax(new_policy)),
            }
        )

    def summarize(
        rows: list[dict[str, float | bool | None]],
    ) -> dict[str, float | int | None]:
        def numeric(key: str) -> list[float]:
            return [float(row[key]) for row in rows if row[key] is not None]

        return {
            "rows": len(rows),
            "top1_agreement": float(np.mean(numeric("top1_agreement"))),
            "mean_l1": float(np.mean(numeric("l1"))),
            "mean_js": float(np.mean(numeric("js"))),
            "mean_old_to_current_cross_entropy": (
                float(np.mean(numeric("old_to_new_ce")))
                if numeric("old_to_new_ce")
                else None
            ),
            "mean_current_to_old_cross_entropy": (
                float(np.mean(numeric("new_to_old_ce")))
                if numeric("new_to_old_ce")
                else None
            ),
            "old_target_entropy": float(np.mean(numeric("old_entropy"))),
            "current_target_entropy": float(np.mean(numeric("current_entropy"))),
            "argmax_changed_fraction": float(np.mean(numeric("argmax_changed"))),
            "substantial_target_mass_movement_fraction": float(
                np.mean(numeric("substantial_mass_movement"))
            ),
            "parent_raw_old_target_top1_agreement": float(
                np.mean(numeric("raw_old_top1"))
            ),
            "parent_raw_refreshed_target_top1_agreement": float(
                np.mean(numeric("raw_current_top1"))
            ),
        }

    all_rows = [row for rows in buckets.values() for row in rows]
    return {
        "overall": summarize(all_rows),
        "by_active_stone_bucket": {
            name: summarize(rows) for name, rows in sorted(buckets.items())
        },
    }
