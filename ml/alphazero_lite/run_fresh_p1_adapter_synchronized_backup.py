#!/usr/bin/env python3
"""Localize post-divergence PUCT Q feedback with frozen P1 backup sequences."""

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
    reconstruct_root_trajectory,
    visit_js,
)
from ml.alphazero_lite.run_fresh_p1_adapter_q_feedback_necessity import (
    C_PUCT,
    SIMULATIONS,
    _seed,
    decode_kalah_v3_base_state,
    first_divergence,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import PUCT

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-frozen-amplified-roots.json"
)
PR222 = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-summary.json"
)
MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
PR220 = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)
LANES = {
    "full": 0,
    "sync_1": 1,
    "sync_8": 8,
    "sync_32": 32,
    "sync_128": 128,
    "sync_rest": None,
}
TRAJECTORY_OFFSETS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 223
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def _init_worker(p1_artifact: str, a16_artifact: str) -> None:
    global _P1, _A16
    _P1, _A16 = (
        ArtifactEvaluator(Path(p1_artifact)),
        ArtifactEvaluator(Path(a16_artifact)),
    )


def _search(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    seed: int,
    *,
    backup_values: list[float] | None = None,
    start: int | None = None,
    width: int | None = None,
) -> tuple[list[dict], dict]:
    trace: list[dict] = []
    backup_hook: Any = None
    if backup_values is not None:
        assert start is not None
        end = SIMULATIONS if width is None else min(SIMULATIONS, start + width - 1)

        def _backup_hook(index: int, raw: float, _trace: dict[str, Any]) -> float:
            return backup_values[index - 1] if start <= index <= end else raw

        backup_hook = _backup_hook

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
        backup_override=backup_hook,
        selection_trace=trace,
        trace_checkpoints={384, 1200},
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return trace, search.root_summary()


def _priors(trace: list[dict]) -> dict[int, float]:
    return {
        int(x["move"]): float(x["prior"])
        for x in trace[0]["selection_path"][0]["children"]
    }


def _trajectory_metric(reference: dict, observed: dict) -> dict[str, Any]:
    actions = sorted(set(reference["q_value"]) | set(observed["q_value"]))
    ref_visits = np.asarray(reference["visit_distribution"], dtype=float)
    obs_visits = np.asarray(observed["visit_distribution"], dtype=float)
    return {
        "root_visit_js": visit_js(ref_visits, obs_visits),
        "root_visit_l1": float(np.abs(ref_visits - obs_visits).sum()),
        "root_q_l1": float(
            sum(
                abs(reference["q_value"].get(a, 0.0) - observed["q_value"].get(a, 0.0))
                for a in actions
            )
        ),
        "q_rank_disagreement": reference["q_ranking"] != observed["q_ranking"],
        "visit_leader_disagreement": reference["visit_leader"]
        != observed["visit_leader"],
        "root_move_disagreement": reference["deterministic_move"]
        != observed["deterministic_move"],
    }


def _sample_trajectory(
    p1: list[dict], full: list[dict], lane: list[dict], d: int
) -> dict[str, Any]:
    p1_t, full_t, lane_t = map(reconstruct_root_trajectory, (p1, full, lane))
    points = sorted(
        {min(SIMULATIONS, d + offset) for offset in TRAJECTORY_OFFSETS} | {384, 1200}
    )
    return {
        str(point): {
            "versus_p1_full": _trajectory_metric(p1_t[point - 1], lane_t[point - 1]),
            "versus_full_a16": _trajectory_metric(full_t[point - 1], lane_t[point - 1]),
        }
        for point in points
    }


def _backup_characterization(
    p1: list[dict], a16: list[dict], d: int
) -> dict[str, dict[str, float | int]]:
    result = {}
    for width in (1, 8, 32, 128):
        pairs = [
            (float(a16[i]["backed_up_value"]), float(p1[i]["backed_up_value"]))
            for i in range(d - 1, min(SIMULATIONS, d - 1 + width))
        ]
        gaps = [left - right for left, right in pairs]
        result[str(width)] = {
            "abs_backup_gap_auc": float(sum(abs(value) for value in gaps)),
            "signed_backup_gap": float(sum(gaps)),
            "opposite_sign_count": sum(left * right < 0 for left, right in pairs),
            "max_abs_backup_gap": float(max(map(abs, gaps), default=0.0)),
        }
    return result


def _lane_invariants(
    full: list[dict],
    lane: list[dict],
    p1_backups: list[float],
    d: int,
    width: int | None,
) -> dict[str, bool]:
    end = SIMULATIONS if width is None else min(SIMULATIONS, d + width - 1)
    before = range(d - 1)
    active = range(d - 1, end)
    return {
        "prefix_selection_matches_full": all(
            lane[i]["selection_path"] == full[i]["selection_path"] for i in before
        ),
        "prefix_backups_match_full": all(
            lane[i]["backed_up_value"] == full[i]["backed_up_value"] for i in before
        ),
        "first_divergence_selection_matches_full": lane[d - 1]["selection_path"]
        == full[d - 1]["selection_path"],
        "requested_p1_backups_applied": all(
            abs(lane[i]["backed_up_value"] - p1_backups[i]) <= 1e-12 for i in active
        ),
        "root_visits_1200": lane[-1]["root_visit_count_after"] == SIMULATIONS,
    }


def _audit(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    state, metadata = task
    assert _P1 is not None and _A16 is not None
    seed = _seed(metadata["state_hash"])
    p1, p1_summary = _search(_P1, state, seed)
    full, full_summary = _search(_A16, state, seed)
    divergence = first_divergence(full, p1)
    if divergence is None or "invariant_failure" in divergence:
        return {**metadata, "invariant_failure": "no valid frozen full divergence"}
    d = int(divergence["simulation"])
    backups = [float(row["backed_up_value"]) for row in p1]
    lanes: dict[str, Any] = {
        "full": {"summary": full_summary, "final_move": full_summary["selected_move"]}
    }
    for name, width in LANES.items():
        if name == "full":
            continue
        trace, summary = _search(
            _A16, state, seed, backup_values=backups, start=d, width=width
        )
        lanes[name] = {
            "summary": summary,
            "final_move": summary["selected_move"],
            "invariants": _lane_invariants(full, trace, backups, d, width),
            "trajectory": _sample_trajectory(p1, full, trace, d),
        }
    return {
        **metadata,
        "search_seed": seed,
        "p1_final_move": p1_summary["selected_move"],
        "full_a16_final_move": full_summary["selected_move"],
        "first_divergence": divergence,
        "p1_backups": backups,
        "p1_root_priors": _priors(p1),
        "a16_root_priors": _priors(full),
        "full_trajectory": _sample_trajectory(p1, full, full, d),
        "backup_characterization": _backup_characterization(p1, full, d),
        "lanes": lanes,
        "invariant_failure": None,
    }


def _bootstrap(values: list[bool]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "samples": BOOTSTRAP_SAMPLES,
    }


def _control_subset(records: list[dict], frozen_records: list[dict]) -> list[dict]:
    """Freeze a greedy pre-intervention match on PR #222 baseline covariates."""
    unused = {
        row["state_hash"]: row
        for row in records
        if row["descriptive_class"] == "washed_out"
    }
    selected = []
    for target in frozen_records:
        target_d = target["actual_first_divergence"]["simulation"]
        target_l1 = target["baseline"]["policy_l1"]
        target_legal = target["legal_move_count"]
        best = min(
            unused.values(),
            key=lambda candidate: (
                abs(candidate["baseline"]["policy_l1"] - target_l1)
                + abs(candidate["actual_first_divergence"]["simulation"] - target_d)
                / SIMULATIONS
                + abs(candidate["legal_move_count"] - target_legal) / 6,
                candidate["state_hash"],
            ),
        )
        selected.append(best)
        del unused[best["state_hash"]]
    return selected


def _classification(
    invariants: dict[str, bool], primary: dict[str, Any], controls: dict[str, Any]
) -> tuple[str, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            "Fix the causal/search contract before interpreting synchronization results.",
        )
    if controls["new_divergences_sync_rest"] >= 12:
        return (
            "intervention_creates_new_instability",
            "Use matched-node Q synchronization instead.",
        )
    rescue = {
        lane: primary[lane]["rescue_rate"]["estimate"]
        for lane in LANES
        if lane != "full"
    }
    if rescue["sync_1"] >= 0.7:
        return (
            "single_backup_shock_seeds_q_cascade",
            "Audit divergent leaf states and value calibration.",
        )
    if max(rescue["sync_8"], rescue["sync_32"]) >= 0.7:
        return (
            "early_backup_window_drives_cascade",
            "Inspect and calibrate early post-divergence leaf-state values.",
        )
    if max(rescue["sync_128"], rescue["sync_rest"]) >= 0.7:
        return (
            "distributed_backup_feedback_required",
            "Test conservative search-time Q/value feedback damping before changing the value head.",
        )
    if rescue["sync_rest"] < 0.5:
        return (
            "backup_sequence_not_sufficient",
            "Synchronize Q statistics on matched tree nodes/actions rather than scalar backup values.",
        )
    return (
        "inconclusive",
        "Audit matched-node Q statistics before changing trained components.",
    )


def _report(summary: dict[str, Any]) -> str:
    rows = [
        "| Root hash | sync_1 | sync_8 | sync_32 | sync_128 | sync_rest | Minimum |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary["primary_records"]:
        rows.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                record["state_hash"],
                *(
                    "rescued"
                    if record["lanes"][lane]["final_move"] == record["p1_final_move"]
                    else "retained"
                    for lane in ("sync_1", "sync_8", "sync_32", "sync_128", "sync_rest")
                ),
                record["minimum_sync_window"] or "never",
            )
        )
    return "\n".join(
        [
            "# Synchronized Backup Localization of Post-Divergence Q Feedback",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "## Primary Outcome",
            "",
            "```json",
            json.dumps(summary["primary_outcome"], indent=2, sort_keys=True),
            "```",
            "",
            "## Minimum Synchronization Window",
            "",
            "```json",
            json.dumps(
                summary["minimum_sync_window_distribution"], indent=2, sort_keys=True
            ),
            "```",
            "",
            "## Frozen 40-Root Retention Table",
            "",
            *rows,
            "",
            "## Invariants",
            "",
            "```json",
            json.dumps(summary["invariants"], indent=2, sort_keys=True),
            "```",
            "",
            "## Negative Controls",
            "",
            "```json",
            json.dumps(summary["negative_controls"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _serialized_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the summary reviewable; complete P1 sequences live in the cache."""
    return {
        **{
            key: value
            for key, value in record.items()
            if key not in {"p1_backups", "lanes"}
        },
        "lanes": {
            name: {key: value for key, value in lane.items() if key != "summary"}
            for name, lane in record["lanes"].items()
        },
    }


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
        / "docs/data/alphazero-lite-fresh-p1-adapter-synchronized-backup-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-synchronized-backup-results.md",
    )
    parser.add_argument(
        "--out-cache",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-synchronized-backup-cache.json",
    )
    args = parser.parse_args()
    frozen, prior, manifest = (
        json.loads(path.read_text()) for path in (FROZEN, PR222, MANIFEST)
    )
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    manifest_by_hash = {row["state_hash"]: row for row in manifest["rows"]}
    frozen_hashes = frozen["full_amplified_1200"]
    control_metadata = _control_subset(
        prior["records"], [prior_by_hash[key] for key in frozen_hashes]
    )
    selected_metadata = [manifest_by_hash[key] for key in frozen_hashes] + [
        manifest_by_hash[row["state_hash"]] for row in control_metadata
    ]
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    tasks = [
        (
            decode_kalah_v3_base_state(list(rows[int(meta["replay_index"])]["state"])),
            meta,
        )
        for meta in selected_metadata
    ]
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(p1_artifact), str(a16_artifact)),
    ) as executor:
        records = list(executor.map(_audit, tasks, chunksize=1))
    by_hash = {row["state_hash"]: row for row in records}
    primary_records = [by_hash[key] for key in frozen_hashes]
    control_hashes = [row["state_hash"] for row in control_metadata]
    controls = [by_hash[key] for key in control_hashes]
    cache = {
        "schema": "azlite_p1_backup_cache_v1",
        "root_order": frozen_hashes,
        "backups": {key: by_hash[key]["p1_backups"] for key in frozen_hashes},
    }
    cache_sha = hashlib.sha256(
        json.dumps(cache, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    primary: dict[str, Any] = {}
    retained_sets = {}
    for lane in LANES:
        if lane == "full":
            continue
        retained = [
            row["state_hash"]
            for row in primary_records
            if row["lanes"][lane]["final_move"] != row["p1_final_move"]
        ]
        retained_sets[lane] = set(retained)
        values = [
            row["state_hash"] not in retained_sets[lane] for row in primary_records
        ]
        primary[lane] = {
            "retention_rate": _bootstrap([not value for value in values]),
            "rescue_rate": _bootstrap(values),
            "retained_root_hashes": retained,
            "rescued_root_hashes": [
                row["state_hash"]
                for row in primary_records
                if row["state_hash"] not in retained_sets[lane]
            ],
        }
    minimum = {
        "rescued_by_1": 0,
        "first_rescued_by_8": 0,
        "first_rescued_by_32": 0,
        "first_rescued_by_128": 0,
        "requires_rest": 0,
        "never_rescued": 0,
    }
    for row in primary_records:
        rescued = [
            lane
            for lane in ("sync_1", "sync_8", "sync_32", "sync_128", "sync_rest")
            if row["lanes"][lane]["final_move"] == row["p1_final_move"]
        ]
        label = (
            "never_rescued"
            if not rescued
            else "rescued_by_1"
            if rescued[0] == "sync_1"
            else f"first_rescued_by_{rescued[0].split('_')[1]}"
            if rescued[0] != "sync_rest"
            else "requires_rest"
        )
        minimum[label] += 1
        row["minimum_sync_window"] = None if not rescued else rescued[0]
    invariants = {
        "frozen_identity_hash": frozen["identity_hash"]
        == "60ebaa037dc931f9d9a7d71e0171c5708db995ae88a119346ffa639812c5b9d1",
        "replay_hash": sha256_file(replay) == prior["hashes"]["replay"],
        "artifact_hashes": sha256_file(p1_artifact / "weights.json")
        == prior["hashes"]["p1_artifact_weights"]
        and sha256_file(a16_artifact / "weights.json")
        == prior["hashes"]["a16_artifact_weights"],
        "state_hashes": all(
            canonical_game_state_hash(KalahGame.from_state(state)) == meta["state_hash"]
            for state, meta in tasks
        ),
        "full_amplified_reproduces": all(
            row["full_a16_final_move"] != row["p1_final_move"]
            for row in primary_records
        ),
        "full_first_divergences_reproduce": all(
            all(
                row["first_divergence"][key]
                == prior_by_hash[row["state_hash"]]["actual_first_divergence"][key]
                for key in ("simulation", "depth", "state_hash", "action_pair")
            )
            for row in primary_records
        ),
        "all_sync_invariants": all(
            all(row["lanes"][lane]["invariants"].values())
            for row in records
            for lane in LANES
            if lane != "full"
        ),
    }
    negative = {
        lane: sum(
            row["lanes"][lane]["final_move"] != row["p1_final_move"] for row in controls
        )
        for lane in ("full", "sync_32", "sync_rest")
    }
    negative["new_divergences_sync_rest"] = negative["sync_rest"] - negative["full"]
    _init_worker(str(p1_artifact), str(a16_artifact))
    held_out = [
        _audit((row["state"], {"state_hash": row["state_hash"]}))
        for row in json.loads(PR220.read_text())["first_game_divergences"]
    ]
    classification, follow_up = _classification(invariants, primary, negative)
    summary = {
        "schema": "azlite_synchronized_backup_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "priors_modified": False,
            "c_puct": C_PUCT,
            "fpu_mode": "zero",
            "simulations": SIMULATIONS,
            "root_noise": False,
        },
        "hashes": {
            "frozen_identity": frozen["identity_hash"],
            "replay": sha256_file(replay),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
            "p1_backup_cache": cache_sha,
        },
        "invariants": invariants,
        "frozen_amplified_roots": frozen_hashes,
        "negative_control_roots": control_hashes,
        "primary_outcome": primary,
        "minimum_sync_window_distribution": minimum,
        "negative_controls": negative,
        "nestedness_violations": {
            f"{later}_not_subset_{earlier}": sorted(
                retained_sets[later] - retained_sets[earlier]
            )
            for earlier, later in zip(
                ("sync_1", "sync_8", "sync_32", "sync_128"),
                ("sync_8", "sync_32", "sync_128", "sync_rest"),
                strict=True,
            )
        },
        "records": [_serialized_record(record) for record in records],
        "primary_records": [_serialized_record(record) for record in primary_records],
        "held_out_pr220": [_serialized_record(record) for record in held_out],
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    for path in (args.out_summary, args.out_report, args.out_cache):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_cache.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
