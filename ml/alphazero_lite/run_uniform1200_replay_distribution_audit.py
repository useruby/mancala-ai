#!/usr/bin/env python3
"""Audit PR #287/#288 replay composition, targets, and forensic failures.

This diagnostic is deliberately read-only with respect to self-play, training,
promotion, and the incumbent artifact.  It only consumes the locked artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    ArtifactEvaluator,
    build_eval_search_options,
    evaluate_artifact_position,
)
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import (  # noqa: E402
    decode_state,
)
from ml.alphazero_lite.forensic_suite import (  # noqa: E402
    canonical_state_key,
    load_suite,
)
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table  # noqa: E402
from ml.alphazero_lite.run_forensic_suite import build_row  # noqa: E402
from ml.alphazero_lite.self_play import value_target_bucket_for_move_index  # noqa: E402

SCHEMA = "azlite_uniform1200_replay_distribution_audit_v1"
SEEDS = (44, 45, 46)
MIN_BUCKET_ROWS = 200
FULL_REPLAY_SHA256 = {
    "44": {
        "control": "3e08d3911139c461c09e2b97c528dea15ffd15393f2bf1bc5b2a8d7da80293ed",
        "uniform1200": "952c0c335d7a828dd71dd4ec6f5aff79d1dce8bd4fc12ac6135f1e6db32f846e",
        "rowmatched": "77a5268982276523591c2e20a37f1c8a78bdf05474744fe2b48dbfe56ee683ca",
    },
    "45": {
        "control": "d914fba65da942ed17e9da706893a53653977d644d5cb4e08441dffff76278d0",
        "uniform1200": "514913e29d646f3a011347ec31a13ac33a2105be48c482e0816e53f92724ba57",
        "rowmatched": "8e3ee111bac3ba15648e78aa340ba9a30d65fad97002db731f2a2eab9ae59eac",
    },
    "46": {
        "control": "6a35f22f454a48a298f13e7fe13a1c4801caa5c804f959e93f3eb1e997851767",
        "uniform1200": "5aa5a4cfdea449b07d40588463bc00115e228026d99a4df62e73eac9e4b44940",
        "rowmatched": "a419f96642365c14c72b0144f6dcf26eb1ec21b389594d7759c37284ff90f9c1",
    },
}
CHECKPOINT_SHA256 = {
    "44": {
        "control": "979f906a80610e2a6648d5bc2ac1a88d47efad75bda53488b045688d12b2c513",
        "uniform1200": "806d04b9452b98dbd7a844e553ea8e4e8af6094cbe37790cf16bc0a0bc90278a",
        "rowmatched": "7aea6ef5e0bcb3caede9a7e88ac9080cac12110d25770c65d605701770fcdd84",
    },
    "45": {
        "control": "52d3df027c0cdccaba7e27f1c2827f673091a610f0103271c77411f0dca4131d",
        "uniform1200": "97a82737efebcee541e9572def4cca08da5e0f5c12728f3a439753ed1b6594da",
        "rowmatched": "c27f24f678b60c768f722d186b286cb197ae111bc75e5342628ffd21f53d5a21",
    },
    "46": {
        "control": "ed61f4a6e943529a5d4fe5d452b00380bad0927c423f983de8fa9a363adc4ee2",
        "uniform1200": "9baa2227602df7ffa0c16c009d16973c67181a079fca810a364b8f4ce97aa604",
        "rowmatched": "3c80a84081ceb43f981056caa528eefa96b914d215be41c453cef263300102de",
    },
}
CATEGORICAL_FEATURES = (
    "phase",
    "legal_move_count",
    "extra_turn_available",
    "capture_available",
    "outcome",
    "current_player",
)
CONTINUOUS_FEATURES = (
    "move_index",
    "player_pit_stones",
    "opponent_pit_stones",
    "active_pit_stones",
    "player_store",
    "opponent_store",
    "signed_store_margin",
    "absolute_store_imbalance",
    "policy_entropy",
    "max_policy_probability",
    "effective_policy_support",
    "value_target",
    "absolute_value_target",
)


def sha256_file(path: Path) -> str:
    """Return a file digest without changing the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy(policy: Iterable[float]) -> float:
    """Return Shannon entropy in bits for a policy distribution."""
    return -sum(float(p) * math.log2(float(p)) for p in policy if float(p) > 0.0)


def total_variation(left: Iterable[float], right: Iterable[float]) -> float:
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def jensen_shannon(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = list(left), list(right)
    midpoint = [(x + y) / 2.0 for x, y in zip(a, b, strict=True)]
    left_kl = sum(
        x * math.log2(x / y) for x, y in zip(a, midpoint, strict=True) if x > 0.0
    )
    right_kl = sum(
        x * math.log2(x / y) for x, y in zip(b, midpoint, strict=True) if x > 0.0
    )
    return (left_kl + right_kl) / 2.0


def _top_action(policy: list[float]) -> int:
    return min(range(len(policy)), key=lambda action: (-policy[action], action))


def replay_features(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Decode one locked replay row and derive only rules-engine features."""
    state = decode_state(list(row["state"]))
    key = canonical_state_key(state)
    game = KalahGame.from_state(state)
    legal_moves = game.possible_moves()
    consequences = move_consequence_table(state)
    policy = [float(value) for value in row["policy"]]
    player = game.current_player
    own = game.pits[player * 6 : (player + 1) * 6]
    opponent = game.pits[(1 - player) * 6 : (2 - player) * 6]
    player_store, opponent_store = (
        game.captured_seeds[player],
        game.captured_seeds[1 - player],
    )
    move_index = int(row.get("move_index", 0))
    feature = {
        "phase": value_target_bucket_for_move_index(move_index),
        "move_index": move_index,
        "current_player": player,
        "outcome": str(row.get("winner")),
        "legal_move_count": len(legal_moves),
        "player_pit_stones": sum(own),
        "opponent_pit_stones": sum(opponent),
        "active_pit_stones": sum(game.pits),
        "player_store": player_store,
        "opponent_store": opponent_store,
        "signed_store_margin": player_store - opponent_store,
        "absolute_store_imbalance": abs(player_store - opponent_store),
        "policy_entropy": entropy(policy),
        "max_policy_probability": max(policy, default=0.0),
        "effective_policy_support": sum(probability > 1e-8 for probability in policy),
        "value_target": float(row["value"]),
        "absolute_value_target": abs(float(row["value"])),
        "extra_turn_available": any(
            item["gives_extra_turn"] for item in consequences if item["legal"]
        ),
        "capture_available": any(
            item["produces_capture"] for item in consequences if item["legal"]
        ),
        "policy": policy,
        "simulations": int(row.get("simulations", 0)),
        "target_telemetry": row.get("target_telemetry"),
    }
    return key, feature


def _empty_state(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": 0,
        "feature": feature,
        "policy_sum": [0.0] * 6,
        "value_sum": 0.0,
        "policy_first": None,
        "value_first": None,
        "policy_varies": False,
        "value_varies": False,
        "simulations": Counter(),
    }


def index_replay(path: Path) -> dict[str, Any]:
    """Build a compact state index plus row and unique-state distributions."""
    states: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            key, feature = replay_features(json.loads(line))
            rows.append(
                {
                    name: feature[name]
                    for name in (*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES)
                }
            )
            entry = states.setdefault(key, _empty_state(feature))
            policy, value = feature["policy"], feature["value_target"]
            entry["count"] += 1
            entry["policy_sum"] = [
                a + b for a, b in zip(entry["policy_sum"], policy, strict=True)
            ]
            entry["value_sum"] += value
            entry["policy_varies"] |= entry["policy_first"] is not None and any(
                abs(a - b) > 1e-9
                for a, b in zip(entry["policy_first"], policy, strict=True)
            )
            entry["value_varies"] |= (
                entry["value_first"] is not None
                and abs(entry["value_first"] - value) > 1e-9
            )
            entry["policy_first"] = (
                policy if entry["policy_first"] is None else entry["policy_first"]
            )
            entry["value_first"] = (
                value if entry["value_first"] is None else entry["value_first"]
            )
            entry["simulations"][feature["simulations"]] += 1
    for entry in states.values():
        entry["policy"] = [value / entry["count"] for value in entry["policy_sum"]]
        entry["value"] = entry["value_sum"] / entry["count"]
    return {"rows": rows, "states": states}


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def _continuous_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.9),
    }


def _ks(left: list[float], right: list[float]) -> float:
    a, b = sorted(left), sorted(right)
    i = j = 0
    maximum = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        maximum = max(maximum, abs(i / len(a) - j / len(b)))
    return maximum


def distribution_delta(
    control: list[dict[str, Any]], uniform: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare distributions with a pre-registered minimum support of 200 rows."""
    categorical: dict[str, Any] = {}
    for name in CATEGORICAL_FEATURES:
        left, right = (
            Counter(str(row[name]) for row in control),
            Counter(str(row[name]) for row in uniform),
        )
        categorical[name] = [
            {
                "bucket": bucket,
                "control_count": left[bucket],
                "uniform_count": right[bucket],
                "control_share": left[bucket] / len(control),
                "uniform_share": right[bucket] / len(uniform),
                "percentage_point_delta": 100
                * (right[bucket] / len(uniform) - left[bucket] / len(control)),
                "meaningful": left[bucket] >= MIN_BUCKET_ROWS
                and right[bucket] >= MIN_BUCKET_ROWS,
            }
            for bucket in sorted(set(left) | set(right))
        ]
    continuous = {}
    for name in CONTINUOUS_FEATURES:
        left, right = (
            [float(row[name]) for row in control],
            [float(row[name]) for row in uniform],
        )
        pooled = math.sqrt(
            (statistics.pvariance(left) + statistics.pvariance(right)) / 2
        )
        continuous[name] = {
            "control": _continuous_summary(left),
            "uniform": _continuous_summary(right),
            "standardized_mean_difference": 0.0
            if pooled == 0
            else (statistics.fmean(right) - statistics.fmean(left)) / pooled,
            "ks": _ks(left, right),
        }
    return {"categorical": categorical, "continuous": continuous}


def overlap(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    left, right = set(control["states"]), set(treatment["states"])
    shared = left & right
    return {
        "control_unique": len(left),
        "uniform_unique": len(right),
        "intersection": len(shared),
        "control_only": len(left - right),
        "uniform_only": len(right - left),
        "jaccard": len(shared) / len(left | right),
        "control_shared_row_fraction": sum(
            control["states"][key]["count"] for key in shared
        )
        / len(control["rows"]),
        "uniform_shared_row_fraction": sum(
            treatment["states"][key]["count"] for key in shared
        )
        / len(treatment["rows"]),
        "control_shared_unique_fraction": len(shared) / len(left),
        "uniform_shared_unique_fraction": len(shared) / len(right),
        "control_multiplicity": Counter(
            entry["count"] for entry in control["states"].values()
        ),
        "uniform_multiplicity": Counter(
            entry["count"] for entry in treatment["states"].values()
        ),
    }


def matched_target_audit(
    control: dict[str, Any], uniform: dict[str, Any]
) -> dict[str, Any]:
    """Compare deterministic per-state target aggregates on canonical intersections."""
    shared = sorted(set(control["states"]) & set(uniform["states"]))
    records = []
    for key in shared:
        left, right = control["states"][key], uniform["states"][key]
        policy_left, policy_right = left["policy"], right["policy"]
        records.append(
            {
                **{name: left["feature"][name] for name in CATEGORICAL_FEATURES},
                "top_action_agreement": _top_action(policy_left)
                == _top_action(policy_right),
                "tv": total_variation(policy_left, policy_right),
                "js": jensen_shannon(policy_left, policy_right),
                "entropy_delta": entropy(policy_right) - entropy(policy_left),
                "max_probability_delta": max(policy_right) - max(policy_left),
                "uniform_probability_control_top": policy_right[
                    _top_action(policy_left)
                ],
                "control_probability_uniform_top": policy_left[
                    _top_action(policy_right)
                ],
                "value_signed_difference": right["value"] - left["value"],
                "value_absolute_difference": abs(right["value"] - left["value"]),
                "value_sign_disagreement": math.copysign(1, right["value"])
                != math.copysign(1, left["value"])
                and right["value"] != 0
                and left["value"] != 0,
                "control_value": left["value"],
                "uniform_value": right["value"],
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"states": 0}
        correlation = (
            0.0
            if len(items) < 2
            else statistics.correlation(
                [row["control_value"] for row in items],
                [row["uniform_value"] for row in items],
            )
        )
        return {
            "states": len(items),
            "top_action_agreement": sum(row["top_action_agreement"] for row in items)
            / len(items),
            "tv": statistics.fmean(row["tv"] for row in items),
            "js": statistics.fmean(row["js"] for row in items),
            "entropy_delta": statistics.fmean(row["entropy_delta"] for row in items),
            "max_probability_delta": statistics.fmean(
                row["max_probability_delta"] for row in items
            ),
            "uniform_probability_control_top": statistics.fmean(
                row["uniform_probability_control_top"] for row in items
            ),
            "control_probability_uniform_top": statistics.fmean(
                row["control_probability_uniform_top"] for row in items
            ),
            "value_signed_difference": statistics.fmean(
                row["value_signed_difference"] for row in items
            ),
            "value_absolute_difference": statistics.fmean(
                row["value_absolute_difference"] for row in items
            ),
            "value_correlation": correlation,
            "value_sign_disagreement": sum(
                row["value_sign_disagreement"] for row in items
            )
            / len(items),
        }

    strata = {"overall": summarize(records)}
    for name in CATEGORICAL_FEATURES:
        for bucket in sorted({str(row[name]) for row in records}):
            selected = [row for row in records if str(row[name]) == bucket]
            if len(selected) >= MIN_BUCKET_ROWS:
                strata[f"{name}={bucket}"] = summarize(selected)
    return {
        "aggregate": strata,
        "duplicate_target_variation": {
            "control_policy_varying_states": sum(
                entry["policy_varies"] for entry in control["states"].values()
            ),
            "uniform_policy_varying_states": sum(
                entry["policy_varies"] for entry in uniform["states"].values()
            ),
            "control_value_varying_states": sum(
                entry["value_varies"] for entry in control["states"].values()
            ),
            "uniform_value_varying_states": sum(
                entry["value_varies"] for entry in uniform["states"].values()
            ),
        },
        "search_telemetry": "root visit/value telemetry was not persisted; stored simulations and target entropy are reported, no MCTS was rerun",
    }


def forensic_change(
    control: dict[str, dict], uniform: dict[str, dict]
) -> dict[str, list[str]]:
    """Classify a matched forensic move as regression, improvement, or neither."""
    regressions, improvements = [], []
    for position_id in sorted(control):
        left, right = control[position_id], uniform[position_id]
        if bool(left["agrees_top1"]) and not bool(right["agrees_top1"]):
            regressions.append(position_id)
        elif (right.get("regret") or 0) > (left.get("regret") or 0):
            regressions.append(position_id)
        elif not bool(left["agrees_top1"]) and bool(right["agrees_top1"]):
            improvements.append(position_id)
        elif (right.get("regret") or 0) < (left.get("regret") or 0):
            improvements.append(position_id)
    return {"regressions": regressions, "improvements": improvements}


def consistent_positions(
    changes: dict[str, dict[str, list[str]]], kind: str
) -> list[str]:
    counts = Counter(
        position for change in changes.values() for position in change[kind]
    )
    return sorted(position for position, count in counts.items() if count >= 2)


def direction_consistency(values: list[float]) -> dict[str, Any]:
    """Summarize an effect direction without treating a p-value as a mechanism."""
    signs = [0 if value == 0 else (1 if value > 0 else -1) for value in values]
    nonzero = [sign for sign in signs if sign]
    return {
        "directions": signs,
        "median_magnitude": statistics.median(abs(value) for value in values),
        "all_three_agree": len(set(nonzero)) == 1 and len(nonzero) == 3,
        "at_least_two_agree": max(signs.count(-1), signs.count(1)) >= 2,
    }


def artifact_paths(root: Path, seed: int) -> dict[str, Path]:
    base = root / "runs" / f"seed{seed}"
    return {
        "control": Path("/tmp/uniform1200-seed-confirmation")
        / "runs"
        / f"seed{seed}"
        / "control_384_192"
        / f"uniform1200-confirm-seed{seed}-control_384_192-iter1",
        "uniform1200": Path("/tmp/uniform1200-seed-confirmation")
        / "runs"
        / f"seed{seed}"
        / "uniform1200"
        / f"uniform1200-confirm-seed{seed}-uniform1200-iter1",
        "rowmatched": base
        / "uniform1200_rowmatched"
        / f"uniform1200-rowmatched-seed{seed}-iter1",
    }


def verify_artifacts(root: Path) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for seed in SEEDS:
        paths = artifact_paths(root, seed)
        replay_paths = {
            "control": paths["control"] / "self_play.jsonl",
            "uniform1200": paths["uniform1200"] / "self_play.jsonl",
            "rowmatched": root / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl",
        }
        verified[str(seed)] = {}
        for lane, path in replay_paths.items():
            actual = sha256_file(path)
            if actual != FULL_REPLAY_SHA256[str(seed)][lane]:
                raise ValueError(f"seed {seed} {lane} replay SHA-256 mismatch")
            checkpoint = paths[lane] / (
                "weights.json" if lane == "rowmatched" else "model.npz"
            )
            actual_checkpoint = sha256_file(checkpoint)
            if actual_checkpoint != CHECKPOINT_SHA256[str(seed)][lane]:
                raise ValueError(f"seed {seed} {lane} checkpoint SHA-256 mismatch")
            verified[str(seed)][lane] = {
                "replay": actual,
                "checkpoint": actual_checkpoint,
            }
    return verified


def evaluate_forensics(root: Path) -> dict[str, Any]:
    """Evaluate all checkpoints once against the same frozen reference artifact."""
    suite = load_suite(
        REPO_ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
    )
    references = json.loads(
        (
            REPO_ROOT
            / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
        ).read_text(encoding="utf-8")
    )
    by_key = {row["canonical_state"]: row for row in references["rows"]}
    artifacts: dict[str, Path] = {
        "incumbent": REPO_ROOT / "storage/ai/alphazero_lite/current"
    }
    for seed in SEEDS:
        for lane, path in artifact_paths(root, seed).items():
            artifacts[f"seed{seed}_{lane}"] = path
    evaluators = {name: ArtifactEvaluator(path) for name, path in artifacts.items()}
    rows: dict[str, dict[str, dict]] = {}
    for name, path in artifacts.items():
        evaluated = {}
        for index, position in enumerate(suite):
            system = evaluate_artifact_position(
                artifact_path=path,
                evaluator=evaluators[name],
                state=position.state,
                simulations=384,
                seed=1042 + index,
                c_puct=1.25,
                search_options=build_eval_search_options(),
            )
            evaluated[position.id] = build_row(
                position=position,
                reference=by_key[position.canonical_key],
                system=system,
            )
        rows[name] = evaluated
    changes = {
        str(seed): forensic_change(
            rows[f"seed{seed}_control"], rows[f"seed{seed}_uniform1200"]
        )
        for seed in SEEDS
    }
    return {
        "reference": "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json",
        "rows": rows,
        "changes": changes,
        "consistent_uniform_regressions": consistent_positions(changes, "regressions"),
        "consistent_uniform_improvements": consistent_positions(
            changes, "improvements"
        ),
    }


def forensic_replay_coverage(root: Path, position_ids: list[str]) -> dict[str, Any]:
    """Retrospectively measure exact forensic-state coverage and descriptors."""
    suite = {
        position.id: position
        for position in load_suite(
            REPO_ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
        )
    }
    wanted = {
        canonical_state_key(suite[position_id].state): position_id
        for position_id in position_ids
    }
    result = {
        position_id: {"descriptor": {}, "lanes": {}} for position_id in position_ids
    }
    for position_id in position_ids:
        state = suite[position_id].state
        game = KalahGame.from_state(state)
        consequences = move_consequence_table(state)
        result[position_id]["descriptor"] = {
            "phase": suite[position_id].phase,
            "legal_move_count": len(game.possible_moves()),
            "active_pit_stones": sum(game.pits),
            "signed_store_margin": game.captured_seeds[game.current_player]
            - game.captured_seeds[1 - game.current_player],
            "extra_turn_available": any(
                item["gives_extra_turn"] for item in consequences if item["legal"]
            ),
            "capture_available": any(
                item["produces_capture"] for item in consequences if item["legal"]
            ),
        }
    for seed in SEEDS:
        paths = artifact_paths(root, seed)
        lane_paths = {
            "control": paths["control"] / "self_play.jsonl",
            "uniform1200": paths["uniform1200"] / "self_play.jsonl",
            "rowmatched": root / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl",
        }
        for lane, path in lane_paths.items():
            exact = Counter()
            signature = Counter()
            total = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    key, feature = replay_features(json.loads(line))
                    total += 1
                    if key in wanted:
                        exact[wanted[key]] += 1
                    for position_id in position_ids:
                        descriptor = result[position_id]["descriptor"]
                        if all(
                            feature[name] == value for name, value in descriptor.items()
                        ):
                            signature[position_id] += 1
            for position_id in position_ids:
                result[position_id]["lanes"][f"seed{seed}_{lane}"] = {
                    "exact_occurrences": exact[position_id],
                    "signature_rows": signature[position_id],
                    "signature_share": signature[position_id] / total,
                }
    return result


def classify(summary: dict[str, Any]) -> tuple[str, str]:
    target_tv = [
        summary["seeds"][str(seed)]["comparisons"]["full_uniform1200"][
            "matched_targets"
        ]["aggregate"]["overall"]["tv"]
        for seed in SEEDS
    ]
    jaccard = [
        summary["seeds"][str(seed)]["comparisons"]["full_uniform1200"]["overlap"][
            "jaccard"
        ]
        for seed in SEEDS
    ]
    if statistics.median(target_tv) >= 0.10 and statistics.median(jaccard) <= 0.10:
        return (
            "uniform1200_distribution_target_interaction",
            "Run one narrowly specified 2x2 ablation in the implicated phase: restore control-like bucket exposure versus uniform exposure, crossed with default versus unsharpened policy targets; keep all replay size, weights, and search settings locked.",
        )
    return (
        "uniform1200_replay_audit_no_clear_mechanism",
        "Expand the frozen forensic/evaluation suite around the repeated failure positions before changing training again.",
    )


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Uniform1200 Replay Distribution And Target-Shift Audit",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "## 1. Verified Artifact SHAs",
        "",
        "| seed | lane | replay | checkpoint |",
        "| ---: | --- | --- | --- |",
    ]
    for seed, lanes in summary["artifact_sha256"].items():
        for lane, digests in lanes.items():
            lines.append(
                f"| {seed} | {lane} | `{digests['replay']}` | `{digests['checkpoint']}` |"
            )
    lines.extend(
        [
            "",
            "## 2. Canonical Overlap",
            "",
            "| seed | comparison | control unique | uniform unique | intersection | Jaccard | shared rows control/uniform |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, comparisons in summary["seeds"].items():
        for name, value in comparisons["comparisons"].items():
            overlap_value = value["overlap"]
            lines.append(
                f"| {seed} | {name} | {overlap_value['control_unique']} | {overlap_value['uniform_unique']} | {overlap_value['intersection']} | {overlap_value['jaccard']:.3f} | {overlap_value['control_shared_row_fraction']:.3f}/{overlap_value['uniform_shared_row_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 3. State-Distribution Delta",
            "",
            f"Pre-registered meaningful subgroup threshold: at least {MIN_BUCKET_ROWS} rows in each lane per seed. Full row- and unique-state-weighted categorical deltas and continuous summaries are in `docs/data/alphazero-lite-uniform1200-replay-distribution-audit.json`.",
            "",
            "## 4. Shared-State Policy-Target Delta",
            "",
            "| seed | comparison | states | top-action agreement | TV | JS | entropy delta |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, comparisons in summary["seeds"].items():
        for name, value in comparisons["comparisons"].items():
            policy = value["matched_targets"]["aggregate"]["overall"]
            lines.append(
                f"| {seed} | {name} | {policy['states']} | {policy['top_action_agreement']:.3f} | {policy['tv']:.3f} | {policy['js']:.3f} | {policy['entropy_delta']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 5. Shared-State Value-Target Delta",
            "",
            "| seed | comparison | mean signed delta | mean absolute delta | correlation | sign disagreement |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, comparisons in summary["seeds"].items():
        for name, value in comparisons["comparisons"].items():
            target = value["matched_targets"]["aggregate"]["overall"]
            lines.append(
                f"| {seed} | {name} | {target['value_signed_difference']:.3f} | {target['value_absolute_difference']:.3f} | {target['value_correlation']:.3f} | {target['value_sign_disagreement']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 6. Per-Seed Forensic Changes",
            "",
            "| seed | uniform regressions vs control | uniform improvements vs control |",
            "| ---: | ---: | ---: |",
        ]
    )
    for seed, change in summary["forensics"].get("changes", {}).items():
        lines.append(
            f"| {seed} | {len(change['regressions'])} | {len(change['improvements'])} |"
        )
    if summary["forensics"].get("rows"):
        lines.extend(
            [
                "",
                "Per-position frozen-reference results (all checkpoints):",
                "",
                "| checkpoint | position | selected | top-1 | regret | blunder | value error | bucket | phase | tags |",
                "| --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- |",
            ]
        )
        for checkpoint, rows in summary["forensics"]["rows"].items():
            for position_id, row in rows.items():
                regret = row.get("regret")
                lines.append(
                    f"| {checkpoint} | {position_id} | {row['selected_move']} | {row['agrees_top1']} | {'-' if regret is None else f'{regret:.4f}'} | {False if regret is None else regret > 0.0} | {row.get('value_error')} | {row['bucket']} | {row['phase']} | {','.join(row['tags'])} |"
                )
    lines.extend(
        [
            "",
            "## 7. Consistent 2/3-Seed Positions",
            "",
            "Consistent regressions: "
            + ", ".join(summary["forensics"]["consistent_uniform_regressions"])
            + ".",
            "",
            "Consistent improvements: "
            + ", ".join(summary["forensics"]["consistent_uniform_improvements"])
            + ".",
            "",
            "## 8. Replay Coverage And Signatures",
            "",
            "Exact canonical coverage and descriptor prevalence for each consistent position are in the machine-readable report. This is retrospective only; no forensic state was added to training.",
            "",
            "## 9. Ranked Candidate Mechanisms",
            "",
            *[f"1. {item}" for item in summary["candidate_mechanisms"]],
            "",
            "## 10. Hard Classification",
            "",
            f"`{summary['classification']}`",
            "",
            "## 11. One Recommended Next Training Experiment",
            "",
            summary["recommendation"],
            "",
            "## Target Provenance",
            "",
            "Stored `policy` is the denoised PUCT root-search visit distribution transformed by the per-ply temperature (`1.0` before the configured threshold, `0.1` after) and then, for these lanes, sharpened by squaring and renormalizing. Gameplay action sampling used Dirichlet noise only in the opening; target Dirichlet epsilon was zero. Stored value is the sharpened outcome-sign/search-magnitude target. Root visit/value telemetry was not persisted, so it is marked unavailable rather than reconstructed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(root: Path, *, with_forensics: bool) -> dict[str, Any]:
    verified = verify_artifacts(root)
    seed_results = {}
    for seed in SEEDS:
        paths = artifact_paths(root, seed)
        control = index_replay(paths["control"] / "self_play.jsonl")
        comparisons = {}
        for name, replay_path in (
            ("full_uniform1200", paths["uniform1200"] / "self_play.jsonl"),
            (
                "rowmatched_uniform1200",
                root / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl",
            ),
        ):
            uniform = index_replay(replay_path)
            comparisons[name] = {
                "overlap": overlap(control, uniform),
                "row_weighted_distribution": distribution_delta(
                    control["rows"], uniform["rows"]
                ),
                "unique_state_weighted_distribution": distribution_delta(
                    [entry["feature"] for entry in control["states"].values()],
                    [entry["feature"] for entry in uniform["states"].values()],
                ),
                "matched_targets": matched_target_audit(control, uniform),
            }
        seed_results[str(seed)] = {"comparisons": comparisons}
    forensics = (
        evaluate_forensics(root)
        if with_forensics
        else {
            "status": "not_requested",
            "consistent_uniform_regressions": [],
            "consistent_uniform_improvements": [],
        }
    )
    if with_forensics:
        forensics["replay_coverage"] = forensic_replay_coverage(
            root, forensics["consistent_uniform_regressions"]
        )
    full_target_tv = [
        seed_results[str(seed)]["comparisons"]["full_uniform1200"]["matched_targets"][
            "aggregate"
        ]["overall"]["tv"]
        for seed in SEEDS
    ]
    full_jaccard = [
        seed_results[str(seed)]["comparisons"]["full_uniform1200"]["overlap"]["jaccard"]
        for seed in SEEDS
    ]
    result = {
        "schema": SCHEMA,
        "promotion": {"performed": False},
        "training": {"performed": False},
        "self_play": {"generated": False},
        "artifact_sha256": verified,
        "minimum_bucket_rows": MIN_BUCKET_ROWS,
        "seeds": seed_results,
        "forensics": forensics,
        "candidate_mechanisms": [
            f"matched-state policy shift: median TV={statistics.median(full_target_tv):.3f} across seeds",
            f"state composition shift: median canonical Jaccard={statistics.median(full_jaccard):.3f} across seeds",
        ],
    }
    result["classification"], result["recommendation"] = classify(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("/home/alex/Mancala/rowmatched-work")
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--skip-forensics", action="store_true")
    parser.add_argument("--resume-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_json:
        if args.skip_forensics:
            raise ValueError("--resume-json cannot be combined with --skip-forensics")
        summary = json.loads(args.resume_json.read_text(encoding="utf-8"))
        if not summary.get("forensics", {}).get("rows"):
            summary["forensics"] = evaluate_forensics(args.artifact_root)
            summary["forensics"]["replay_coverage"] = forensic_replay_coverage(
                args.artifact_root,
                summary["forensics"]["consistent_uniform_regressions"],
            )
        summary["classification"], summary["recommendation"] = classify(summary)
    else:
        summary = run(args.artifact_root, with_forensics=not args.skip_forensics)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
