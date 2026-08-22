#!/usr/bin/env python3
"""Causally ablate value/Q feedback on the frozen PR #221/#222 root population."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.postdivergence_amplification import (
    paired_postdivergence_metrics,
    reconstruct_root_trajectory,
    validate_final_root_trajectory,
    visit_js,
)
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.kalah_rules import KalahGame

REPO_ROOT = Path(__file__).resolve().parents[2]
PR221_SUMMARY = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-summary.json"
)
PR221_MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
PR222_SUMMARY = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-summary.json"
)
PR220_SUMMARY = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)
PR221_MANIFEST_SHA = "20b0aa432b1ee978c9da22d030a6e51e1cc1753989cfd6bf6957129915b22078"
CHECKPOINTS = (384, 1200)
TRAJECTORY_OFFSETS = (1, 4, 16, 32, 64, 128)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 220
C_PUCT = 1.25
SIMULATIONS = 1200
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load the frozen replay without importing the training runner."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    """Return the artifact identity used by the frozen PR #222 contracts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_kalah_v3_base_state(features: list[Any]) -> dict[str, Any]:
    """Recover the frozen replay state from its invertible Kalah-v3 prefix."""

    def stone(index: int) -> int:
        value = float(features[index]) * 48.0
        rounded = round(value)
        if abs(value - rounded) > 1e-5 or rounded < 0:
            raise ValueError("kalah_v3 base feature is not an exact /48 value")
        return int(rounded)

    player = int(round(float(features[14])))
    if len(features) < 15 or player not in (0, 1):
        raise ValueError("invalid kalah_v3 base state")
    return {
        "player_pits": [stone(index) for index in range(6)],
        "opponent_pits": [stone(index) for index in range(6, 12)],
        "player_store": stone(12),
        "opponent_store": stone(13),
        "current_player": player,
    }


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"pr220-margin-sensitivity:{state_hash}".encode()).hexdigest()[
            :16
        ],
        16,
    )


def _same_statistics(left: dict, right: dict) -> bool:
    left_children = {entry["move"]: entry for entry in left["children"]}
    right_children = {entry["move"]: entry for entry in right["children"]}
    return left_children.keys() == right_children.keys() and all(
        left_children[move]["visit_count"] == right_children[move]["visit_count"]
        and abs(left_children[move]["q_value"] - right_children[move]["q_value"])
        <= 1e-12
        and abs(
            left_children[move]["q_component"] - right_children[move]["q_component"]
        )
        <= 1e-12
        for move in left_children
    )


def first_divergence(
    a16_trace: list[dict], p1_trace: list[dict]
) -> dict[str, Any] | None:
    """Reproduce PR #222's actual first selection-path divergence definition."""
    for actual, parent in zip(a16_trace, p1_trace, strict=True):
        for candidate, baseline in zip(
            actual["selection_path"], parent["selection_path"], strict=True
        ):
            if candidate["state_hash"] != baseline["state_hash"]:
                return {
                    "invariant_failure": "path state mismatch before action divergence"
                }
            if not _same_statistics(candidate, baseline):
                return {
                    "invariant_failure": "visits or Q values differ before action divergence"
                }
            if candidate["chosen_move"] != baseline["chosen_move"]:
                return {
                    "simulation": int(actual["simulation_index"]),
                    "depth": int(candidate["tree_depth"]),
                    "state_hash": candidate["state_hash"],
                    "action_pair": [
                        int(candidate["chosen_move"]),
                        int(baseline["chosen_move"]),
                    ],
                }
        if abs(actual["backed_up_value"] - parent["backed_up_value"]) > 1e-12:
            return {"invariant_failure": "backup differs before selection divergence"}
    return None


def _init_worker(p1_artifact: str, a16_artifact: str) -> None:
    global _P1, _A16
    _P1, _A16 = (
        ArtifactEvaluator(Path(p1_artifact)),
        ArtifactEvaluator(Path(a16_artifact)),
    )


def _search(
    evaluator: ArtifactEvaluator, state: dict[str, Any], seed: int, mode: str
) -> tuple[list[dict], dict]:
    trace: list[dict] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        rng=random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        ablation_mode=mode,
        selection_trace=trace,
        trace_checkpoints=set(CHECKPOINTS),
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return trace, search.root_summary()


def _pair_snapshot(p: dict, a: dict) -> dict[str, Any]:
    p_visits, a_visits = p["visit_distribution"], a["visit_distribution"]
    return {
        "p1_root_move": p["deterministic_move"],
        "a16_root_move": a["deterministic_move"],
        "root_move_difference": p["deterministic_move"] != a["deterministic_move"],
        "visit_js": visit_js(p_visits, a_visits),
        "visit_l1": float(np.abs(np.asarray(p_visits) - np.asarray(a_visits)).sum()),
        "p1_visit_leader": p["visit_leader"],
        "a16_visit_leader": a["visit_leader"],
        "visit_leader_difference": p["visit_leader"] != a["visit_leader"],
        "p1_top1_top2_visit_margin": p["top1_top2_visit_margin"],
        "a16_top1_top2_visit_margin": a["top1_top2_visit_margin"],
    }


def _root_priors(trace: list[dict]) -> dict[int, float]:
    return {
        int(row["move"]): float(row["prior"])
        for row in trace[0]["selection_path"][0]["children"]
    }


def _condition(state: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    assert _P1 is not None and _A16 is not None
    p_trace, p_summary = _search(_P1, state, seed, mode)
    a_trace, a_summary = _search(_A16, state, seed, mode)
    p_trajectory, a_trajectory = (
        reconstruct_root_trajectory(p_trace),
        reconstruct_root_trajectory(a_trace),
    )
    divergence = first_divergence(a_trace, p_trace)
    failure = None if divergence is None else divergence.get("invariant_failure")
    first = (
        None
        if divergence is None or failure is not None
        else {
            key: divergence[key]
            for key in ("simulation", "depth", "state_hash", "action_pair")
        }
    )
    result = {
        "invariant_failure": failure,
        "first_divergence": first,
        "p1_root_priors": _root_priors(p_trace),
        "a16_root_priors": _root_priors(a_trace),
        "trajectory_matches_summary": {
            "p1": validate_final_root_trajectory(p_trajectory, p_summary),
            "a16": validate_final_root_trajectory(a_trajectory, a_summary),
        },
        "at_384": _pair_snapshot(p_trajectory[383], a_trajectory[383]),
        "at_1200": _pair_snapshot(p_trajectory[-1], a_trajectory[-1]),
        "p1_backed_up_value_range": p_summary["backed_up_value_range"],
        "a16_backed_up_value_range": a_summary["backed_up_value_range"],
    }
    if first is not None:
        metrics = paired_postdivergence_metrics(
            p_trace, a_trace, p_trajectory, a_trajectory, first["simulation"]
        )
        result["postdivergence_at_384"] = metrics.pop("at_384")
        result.update(metrics)
    if mode == "policy_only":
        result["all_q_components_zero"] = all(
            abs(float(child["q_value"])) <= 1e-12
            for summary in (p_summary, a_summary)
            for child in summary["child_stats"]
        ) and all(
            abs(float(entry["q_component"])) <= 1e-12
            for trace in (p_trace, a_trace)
            for record in trace
            for decision in record["selection_path"]
            for entry in decision["children"]
        )
        result["all_backed_up_values_zero"] = all(
            abs(float(record["backed_up_value"])) <= 1e-12
            for trace in (p_trace, a_trace)
            for record in trace
        )
    if mode == "value_only":
        result["traces_identical"] = p_trace == a_trace
    return result


def _audit_root(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    state, metadata = task
    seed = _seed(metadata["state_hash"])
    return {
        **metadata,
        "search_seed": seed,
        "conditions": {
            mode: _condition(state, seed, mode)
            for mode in ("full", "policy_only", "value_only")
        },
    }


def _interval(values: np.ndarray) -> dict[str, Any]:
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "samples": len(values),
    }


def _bootstrap(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return _interval(np.mean(array[draws], axis=1))


def _bootstrap_distribution(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return {
        "mean": _interval(np.mean(array[draws], axis=1)),
        "median": _interval(np.median(array[draws], axis=1)),
    }


def _event_comparison(full: dict, policy: dict) -> dict[str, Any]:
    left, right = full["first_divergence"], policy["first_divergence"]
    if left is None or right is None:
        return {"both_present": left is not None and right is not None}
    return {
        "both_present": True,
        "same_simulation": left["simulation"] == right["simulation"],
        "same_depth": left["depth"] == right["depth"],
        "same_action_pair": left["action_pair"] == right["action_pair"],
        "simulation_delta": policy["first_divergence"]["simulation"]
        - full["first_divergence"]["simulation"],
    }


def _visit_js_after_divergence(condition: dict[str, Any]) -> dict[str, float] | None:
    first = condition["first_divergence"]
    if first is None:
        return None
    samples = condition.get("windows", {})
    return {
        f"d_plus_{offset}": float(samples[str(offset)]["visit_js"])
        for offset in TRAJECTORY_OFFSETS
        if str(offset) in samples
    }


def _pure_visit_dynamics(record: dict[str, Any]) -> dict[str, Any] | None:
    condition = record["conditions"]["policy_only"]
    if not condition["at_1200"]["root_move_difference"]:
        return None
    lead = condition.get("lead_lag", {})
    first = condition["first_divergence"]
    root_move = lead.get("root_move_difference", {}).get(
        "first_final_persistent_relative_simulation"
    )
    leader = lead.get("visit_leader_difference", {}).get(
        "first_final_persistent_relative_simulation"
    )
    return {
        "state_hash": record["state_hash"],
        "first_divergence": first,
        "first_persistent_visit_leader_difference_relative_simulation": leader,
        "lag_to_final_root_move_relative_simulation": root_move,
        "path_divergence_fraction": lead.get("path_divergence", {}).get(
            "fraction_remaining"
        ),
        "visit_js_trajectory": {
            **(_visit_js_after_divergence(condition) or {}),
            "at_384": condition["at_384"]["visit_js"],
            "at_1200": condition["at_1200"]["visit_js"],
        },
    }


def _classification(
    invariants: dict[str, bool], retention: dict, overlap: dict, full_delta: dict | None
) -> tuple[str, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            "Fix the failed ablation or reproduction contract before interpreting causal effects.",
        )
    if (
        retention["upper_95"] < 0.5
        and full_delta is not None
        and full_delta["mean"]["estimate"] > 0
    ):
        return (
            "q_feedback_necessary_for_amplification",
            "Add a diagnostic synchronized-backup or Q-feedback intervention on the frozen amplified roots to localize which backup differences cause persistence.",
        )
    if retention["lower_95"] > 0.5 and overlap["jaccard"] >= 0.5:
        return (
            "visit_reinforcement_sufficient",
            "Trace visit/U reinforcement and test a causal prior-decay or visit-statistic counterfactual after the first divergence.",
        )
    if retention["lower_95"] <= 0.5 <= retention["upper_95"]:
        return (
            "q_and_visit_both_contribute",
            "Run separate causal interventions on the frozen rescued and retained subsets.",
        )
    if overlap["policy_only_count"] and overlap["jaccard"] < 0.25:
        return (
            "policy_only_changes_failure_population",
            "Implement a more surgical post-divergence Q-feedback intervention instead of global value removal.",
        )
    return (
        "inconclusive",
        "Add a diagnostic synchronized-backup or Q-feedback intervention on the frozen amplified roots to localize which backup differences cause persistence.",
    )


def _report(summary: dict) -> str:
    causal = summary["causal_analysis"]
    return "\n".join(
        [
            "# Q-Feedback Necessity for Post-Divergence Amplification",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "## Frozen Full Outcome",
            "",
            f"The frozen PR #222 full-amplified set contains {len(summary['frozen_amplified_1200'])} roots.",
            "",
            "## Primary Causal Metric",
            "",
            "```json",
            json.dumps(causal["policy_only_retention_1200"], indent=2, sort_keys=True),
            "```",
            "",
            "## Root Identity Overlap",
            "",
            "```json",
            json.dumps(causal["overlap_1200"], indent=2, sort_keys=True),
            "```",
            "",
            "## First-Divergence Timing",
            "",
            "```json",
            json.dumps(causal["first_divergence_timing"], indent=2, sort_keys=True),
            "```",
            "",
            "## Visit-JS Causal Effect",
            "",
            "```json",
            json.dumps(causal["delta_visit_js"], indent=2, sort_keys=True),
            "```",
            "",
            "## Pure Visit Reinforcement",
            "",
            f"Policy-only retained roots: {len(causal['policy_only_pure_visit_dynamics'])}.",
            "",
            "## Held-Out PR #220 States",
            "",
            "```json",
            json.dumps(summary["held_out_pr220"], indent=2, sort_keys=True),
            "```",
            "",
            "## Invariants",
            "",
            "```json",
            json.dumps(summary["invariants"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-results.md",
    )
    parser.add_argument(
        "--out-frozen-roots",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-frozen-amplified-roots.json",
    )
    args = parser.parse_args()
    pr221, manifest, pr222 = (
        json.loads(path.read_text())
        for path in (PR221_SUMMARY, PR221_MANIFEST, PR222_SUMMARY)
    )
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    tasks = [
        (
            decode_kalah_v3_base_state(
                list(rows[int(metadata["replay_index"])]["state"])
            ),
            metadata,
        )
        for metadata in manifest["rows"]
    ]
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(p1_artifact), str(a16_artifact)),
    ) as executor:
        records = list(executor.map(_audit_root, tasks, chunksize=1))
    old_by_hash = {record["state_hash"]: record for record in pr222["records"]}
    frozen = sorted(
        record["state_hash"]
        for record in pr222["records"]
        if record["root_move_difference_1200"]
    )
    frozen_late = sorted(
        record["state_hash"]
        for record in pr222["records"]
        if not record["root_move_difference_384"]
        and record["root_move_difference_1200"]
    )
    washed = sorted(
        record["state_hash"]
        for record in pr222["records"]
        if record["descriptive_class"] == "washed_out"
    )
    by_hash = {record["state_hash"]: record for record in records}
    full_amplified = {
        key
        for key, record in by_hash.items()
        if record["conditions"]["full"]["at_1200"]["root_move_difference"]
    }
    policy_amplified = {
        key
        for key, record in by_hash.items()
        if record["conditions"]["policy_only"]["at_1200"]["root_move_difference"]
    }
    full_384 = {
        key
        for key, record in by_hash.items()
        if record["conditions"]["full"]["at_384"]["root_move_difference"]
    }
    policy_384 = {
        key
        for key, record in by_hash.items()
        if record["conditions"]["policy_only"]["at_384"]["root_move_difference"]
    }
    retained = sorted(set(frozen) & policy_amplified)
    rescued = sorted(set(frozen) - policy_amplified)
    comparisons = [
        _event_comparison(
            record["conditions"]["full"], record["conditions"]["policy_only"]
        )
        for record in records
    ]

    def delta(keys: set[str], budget: int) -> list[float]:
        return [
            record["conditions"]["full"][f"at_{budget}"]["visit_js"]
            - record["conditions"]["policy_only"][f"at_{budget}"]["visit_js"]
            for record in records
            if record["state_hash"] in keys
        ]

    def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
        intersection = left & right
        return {
            "full_count": len(left),
            "policy_only_count": len(right),
            "intersection_count": len(intersection),
            "full_only_count": len(left - right),
            "policy_only_only_count": len(right - left),
            "jaccard": len(intersection) / len(left | right) if left | right else 1.0,
        }

    invariants = {
        "pr221_manifest_hash_and_order": sha256_file(PR221_MANIFEST)
        == PR221_MANIFEST_SHA
        and all(
            canonical_game_state_hash(KalahGame.from_state(state))
            == metadata["state_hash"]
            for state, metadata in tasks
        ),
        "replay_hash": sha256_file(replay) == pr222["hashes"]["replay"],
        "artifact_hashes": sha256_file(p1_artifact / "weights.json")
        == pr222["hashes"]["p1_artifact_weights"]
        and sha256_file(a16_artifact / "weights.json")
        == pr222["hashes"]["a16_artifact_weights"],
        "full_first_divergences_reproduce": all(
            record["conditions"]["full"]["first_divergence"]
            == (
                None
                if old_by_hash[record["state_hash"]]["actual_first_divergence"] is None
                else {
                    key: old_by_hash[record["state_hash"]]["actual_first_divergence"][
                        key
                    ]
                    for key in ("simulation", "depth", "state_hash", "action_pair")
                }
            )
            for record in records
        ),
        "no_first_divergence_invariant_failure": all(
            condition["invariant_failure"] is None
            for record in records
            for condition in record["conditions"].values()
        ),
        "full_amplified_1200_reproduces": full_amplified == set(frozen),
        "full_visit_js_reproduces": all(
            abs(
                record["conditions"]["full"]["at_1200"]["visit_js"]
                - old_by_hash[record["state_hash"]]["visit_js_1200"]
            )
            <= 1e-12
            for record in records
        ),
        "trajectory_matches_puct_summary": all(
            all(condition["trajectory_matches_summary"].values())
            for record in records
            for condition in record["conditions"].values()
        ),
        "policy_only_zero_q_and_backups": all(
            condition["all_q_components_zero"]
            and condition["all_backed_up_values_zero"]
            for record in records
            for condition in [record["conditions"]["policy_only"]]
        ),
        "policy_only_no_q_rank_divergence": all(
            not any(
                sample["q_rank_disagreement"]
                for sample in condition.get("windows", {}).values()
            )
            for record in records
            for condition in [record["conditions"]["policy_only"]]
        ),
        "policy_only_root_priors_match_full": all(
            condition["p1_root_priors"] == full["p1_root_priors"]
            and condition["a16_root_priors"] == full["a16_root_priors"]
            for record in records
            for condition, full in [
                (record["conditions"]["policy_only"], record["conditions"]["full"])
            ]
        ),
        "policy_only_prior_l1_matches_full": all(
            abs(
                sum(
                    abs(
                        record["conditions"]["policy_only"]["p1_root_priors"][move]
                        - record["conditions"]["policy_only"]["a16_root_priors"][move]
                    )
                    for move in record["conditions"]["full"]["p1_root_priors"]
                )
                - old_by_hash[record["state_hash"]]["baseline"]["policy_l1"]
            )
            <= 1e-9
            for record in records
        ),
        "value_only_flat_identity": all(
            record["conditions"]["value_only"]["p1_root_priors"]
            == record["conditions"]["value_only"]["a16_root_priors"]
            and record["conditions"]["value_only"]["traces_identical"]
            and not record["conditions"]["value_only"]["at_1200"][
                "root_move_difference"
            ]
            for record in records
        ),
    }
    retention = _bootstrap(
        [
            float(
                by_hash[key]["conditions"]["policy_only"]["at_1200"][
                    "root_move_difference"
                ]
            )
            for key in frozen
        ]
    )
    retention_384 = _bootstrap(
        [
            float(
                by_hash[key]["conditions"]["policy_only"]["at_384"][
                    "root_move_difference"
                ]
            )
            for key in frozen
        ]
    )
    overlap_1200 = overlap(full_amplified, policy_amplified)
    causal_delta = {
        group: {
            str(budget): _bootstrap_distribution(delta(keys, budget))
            for budget in CHECKPOINTS
        }
        for group, keys in {
            "full_amplified": set(frozen),
            "washed_out": set(washed),
            "all_roots": set(by_hash),
        }.items()
    }
    classification, follow_up = _classification(
        invariants, retention, overlap_1200, causal_delta["full_amplified"]["1200"]
    )
    _init_worker(str(p1_artifact), str(a16_artifact))
    held_out = []
    for row in {
        item["state_hash"]: item
        for item in json.loads(PR220_SUMMARY.read_text())["first_game_divergences"]
    }.values():
        seed = _seed(row["state_hash"])
        full, policy = (
            _condition(row["state"], seed, mode) for mode in ("full", "policy_only")
        )
        held_out.append(
            {
                "state_hash": row["state_hash"],
                "full": {
                    "first_divergence": full["first_divergence"],
                    "at_384": full["at_384"],
                    "at_1200": full["at_1200"],
                },
                "policy_only": {
                    "first_divergence": policy["first_divergence"],
                    "at_384": policy["at_384"],
                    "at_1200": policy["at_1200"],
                },
                "removing_q_rescues_p1_root_move": full["at_1200"][
                    "root_move_difference"
                ]
                and not policy["at_1200"]["root_move_difference"],
            }
        )
    frozen_payload = {
        "source_pr": 222,
        "full_amplified_1200": frozen,
        "full_late_amplified": frozen_late,
        "washed_out": washed,
    }
    frozen_payload["identity_hash"] = hashlib.sha256(
        json.dumps(frozen_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reported_root_hashes = set(frozen) | policy_amplified
    summary = {
        "schema": "azlite_pr222_q_feedback_necessity_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "c_puct": C_PUCT,
            "simulations": SIMULATIONS,
            "fpu_mode": "zero",
            "root_noise": False,
            "root_selection": "deterministic",
        },
        "hashes": {
            "manifest": sha256_file(PR221_MANIFEST),
            "replay": sha256_file(replay),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
            "pr220_commit": pr222["hashes"]["pr220_commit"],
        },
        "invariants": invariants,
        "frozen_amplified_1200": frozen,
        "frozen_late_amplified": frozen_late,
        "frozen_washed_out": washed,
        "frozen_identity_hash": frozen_payload["identity_hash"],
        "reported_root_records": [
            record for record in records if record["state_hash"] in reported_root_hashes
        ],
        "causal_analysis": {
            "policy_only_retention_1200": retention,
            "policy_only_retention_384": retention_384,
            "overlap_1200": overlap_1200,
            "overlap_384": overlap(full_384, policy_384),
            "first_divergence_timing": {
                "same_first_simulation_rate": float(
                    np.mean([row.get("same_simulation", False) for row in comparisons])
                ),
                "same_depth_rate": float(
                    np.mean([row.get("same_depth", False) for row in comparisons])
                ),
                "same_action_pair_rate": float(
                    np.mean([row.get("same_action_pair", False) for row in comparisons])
                ),
                "simulation_delta": _bootstrap(
                    [
                        row["simulation_delta"]
                        for row in comparisons
                        if row.get("both_present")
                    ]
                ),
            },
            "delta_visit_js": causal_delta,
            "retained": retained,
            "rescued": rescued,
            "policy_only_pure_visit_dynamics": [
                item
                for record in records
                if (item := _pure_visit_dynamics(record)) is not None
            ],
            "full_metrics_retained_vs_rescued": {
                group: {
                    metric: _bootstrap_distribution(
                        [
                            old_by_hash[key]["early_metrics"][metric]
                            if metric != "policy_l1"
                            else old_by_hash[key]["baseline"][metric]
                            for key in keys
                        ]
                    )
                    for metric in (
                        "q_divergence_auc_32",
                        "backup_gap_auc_32",
                        "visit_js_auc_32",
                        "path_divergence_fraction_32",
                        "policy_l1",
                    )
                }
                for group, keys in {"retained": retained, "rescued": rescued}.items()
            },
        },
        "held_out_pr220": held_out,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    for path in (args.out_summary, args.out_report, args.out_frozen_roots):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_frozen_roots.write_text(
        json.dumps(frozen_payload, indent=2, sort_keys=True) + "\n"
    )
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
