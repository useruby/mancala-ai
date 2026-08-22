#!/usr/bin/env python3
"""Localize post-divergence feedback with equivalent P1 selection-Q snapshots."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
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
from ml.alphazero_lite.run_fresh_p1_adapter_synchronized_backup import _control_subset
from ml.alphazero_lite.self_play import Node, PUCT

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
LANES = ("full", "root_qsync", "all_qsync_32", "all_qsync_rest")
TRAJECTORY_OFFSETS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 224
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
    pre_simulation_hook: Any = None,
    selection_q_override: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace: list[dict[str, Any]] = []
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
        pre_simulation_hook=pre_simulation_hook,
        selection_q_override=selection_q_override,
        selection_trace=trace,
        trace_checkpoints={384, 1200},
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return trace, search.root_summary()


def _game_hash(game: KalahGame) -> str:
    encoded = json.dumps(
        game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _walk(root: Node) -> list[Node]:
    nodes: list[Node] = []
    pending = [root]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        pending.extend(node.children.values())
    return nodes


def _snapshot(node: Node) -> dict[str, Any]:
    children = {
        int(move): {
            "visit_count": int(child.visit_count),
            "q_value": float(child.q_value),
        }
        for move, child in node.children.items()
    }
    return {
        "player_to_move": int(node.game.current_player),
        "children": children,
        "signature": tuple(
            (move, row["visit_count"], row["q_value"])
            for move, row in sorted(children.items())
        ),
    }


def _reference_snapshots(
    eligible_hashes: set[str], telemetry: Counter[str]
) -> tuple[dict[int, dict[str, dict[str, Any]]], Any]:
    """Capture only preflight-visible states; reject ambiguous repeated state hashes."""
    snapshots: dict[int, dict[str, dict[str, Any]]] = {}

    def hook(simulation: int, root: Node) -> None:
        candidates: dict[str, list[dict[str, Any]]] = {}
        for node in _walk(root):
            state_hash = _game_hash(node.game)
            if state_hash in eligible_hashes:
                candidates.setdefault(state_hash, []).append(_snapshot(node))
        rows: dict[str, dict[str, Any]] = {}
        for state_hash, copies in candidates.items():
            telemetry["candidate_nodes"] += len(copies)
            if len(copies) > 1:
                telemetry["duplicate_state_hashes"] += 1
            if len({row["signature"] for row in copies}) != 1:
                telemetry["non_equivalent_duplicate_state_hashes"] += 1
                continue
            rows[state_hash] = copies[0]
            telemetry["reference_snapshots"] += 1
        snapshots[simulation] = rows

    return snapshots, hook


def _q_override(
    references: dict[int, dict[str, dict[str, Any]]],
    telemetry: Counter[str],
    *,
    start: int,
    end: int,
    root_hash: str | None,
) -> Any:
    def override(
        simulation: int, state_hash: str, move: int, raw_q: float, visits: int
    ) -> float | None:
        if simulation < start or simulation > end:
            return None
        if root_hash is not None and state_hash != root_hash:
            return None
        telemetry["eligible_selection_edges"] += 1
        reference = references.get(simulation, {}).get(state_hash)
        if reference is None:
            telemetry["missing_reference"] += 1
            return None
        child = reference["children"].get(move)
        if child is None:
            telemetry["missing_reference_action"] += 1
            return None
        if int(visits) <= 0:
            telemetry["a16_unvisited_skips"] += 1
            return None
        if child["visit_count"] <= 0:
            telemetry["p1_unvisited_skips"] += 1
            return None
        telemetry["applied_selection_q"] += 1
        return float(child["q_value"])

    return override


def _trajectory_metric(
    reference: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    actions = sorted(set(reference["q_value"]) | set(observed["q_value"]))
    return {
        "root_visit_js": visit_js(
            np.asarray(reference["visit_distribution"]),
            np.asarray(observed["visit_distribution"]),
        ),
        "root_q_l1": float(
            sum(
                abs(reference["q_value"].get(a, 0.0) - observed["q_value"].get(a, 0.0))
                for a in actions
            )
        ),
        "root_move_disagreement": reference["deterministic_move"]
        != observed["deterministic_move"],
    }


def _trajectories(
    p1: list[dict[str, Any]],
    full: list[dict[str, Any]],
    lane: list[dict[str, Any]],
    d: int,
) -> dict[str, Any]:
    p1_t, full_t, lane_t = map(reconstruct_root_trajectory, (p1, full, lane))
    points = sorted(
        {min(SIMULATIONS, d + offset) for offset in TRAJECTORY_OFFSETS} | {384, 1200}
    )
    return {
        str(point): {
            "versus_p1": _trajectory_metric(p1_t[point - 1], lane_t[point - 1]),
            "versus_full_a16": _trajectory_metric(full_t[point - 1], lane_t[point - 1]),
        }
        for point in points
    }


def _lane_invariants(
    full: list[dict[str, Any]], lane: list[dict[str, Any]], d: int
) -> dict[str, bool]:
    prefix = range(d)  # The intervention begins before simulation d + 1.
    overridden = [
        entry
        for record in lane
        for node in record["selection_path"]
        for entry in node["children"]
        if entry.get("selection_q_overridden")
    ]
    return {
        "prefix_selection_matches_full_through_d": all(
            lane[i]["selection_path"] == full[i]["selection_path"] for i in prefix
        ),
        "prefix_backups_match_full_through_d": all(
            lane[i]["backed_up_value"] == full[i]["backed_up_value"] for i in prefix
        ),
        # An override can equal the local Q; stored Q must remain untouched either way.
        "selection_q_only": all(
            entry["stored_q_value"] == entry["q_value"] for entry in overridden
        ),
        "root_visits_1200": lane[-1]["root_visit_count_after"] == SIMULATIONS,
    }


def _audit(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    state, metadata = task
    assert _P1 is not None and _A16 is not None
    seed = _seed(metadata["state_hash"])
    full, full_summary = _search(_A16, state, seed)
    eligible = {
        node["state_hash"] for record in full for node in record["selection_path"]
    }
    capture_telemetry: Counter[str] = Counter()
    references, hook = _reference_snapshots(eligible, capture_telemetry)
    p1, p1_summary = _search(_P1, state, seed, pre_simulation_hook=hook)
    divergence = first_divergence(full, p1)
    if divergence is None or "invariant_failure" in divergence:
        return {**metadata, "invariant_failure": "no valid frozen full divergence"}
    d = int(divergence["simulation"])
    root_hash = _game_hash(KalahGame.from_state(state))
    lanes: dict[str, Any] = {
        "full": {"summary": full_summary, "final_move": full_summary["selected_move"]}
    }
    lane_specs = {
        "root_qsync": (root_hash, SIMULATIONS),
        "all_qsync_32": (None, min(SIMULATIONS, d + 32)),
        "all_qsync_rest": (None, SIMULATIONS),
    }
    for name, (only_root, end) in lane_specs.items():
        telemetry: Counter[str] = Counter()
        trace, summary = _search(
            _A16,
            state,
            seed,
            selection_q_override=_q_override(
                references, telemetry, start=d + 1, end=end, root_hash=only_root
            ),
        )
        lanes[name] = {
            "summary": summary,
            "final_move": summary["selected_move"],
            "invariants": _lane_invariants(full, trace, d),
            "coverage_telemetry": dict(telemetry),
            "trajectories_requested": _trajectories(p1, full, trace, d),
        }
    return {
        **metadata,
        "search_seed": seed,
        "p1_final_move": p1_summary["selected_move"],
        "full_a16_final_move": full_summary["selected_move"],
        "first_divergence": divergence,
        "preflight_full_a16_state_hash_count": len(eligible),
        "reference_capture_telemetry": dict(capture_telemetry),
        "reference_cache_sha": hashlib.sha256(
            json.dumps(references, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "full_trajectories_requested": _trajectories(p1, full, full, d),
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


def _classification(
    invariants: dict[str, bool], primary: dict[str, Any], controls: dict[str, int]
) -> tuple[str, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            "Repair the matched-snapshot contract before interpreting selection-Q effects.",
        )
    if controls["new_divergences_all_qsync_rest"] / 40 > 0.1:
        return (
            "matched_q_creates_new_instability",
            "Audit snapshot equivalence and restrict the intervention to matched post-divergence nodes.",
        )
    if primary["root_qsync"]["rescue_rate"]["estimate"] >= 0.7:
        return (
            "root_q_feedback_dominant",
            "Test root-Q confidence/shrinkage rather than modifying the value network.",
        )
    if primary["all_qsync_rest"]["rescue_rate"]["estimate"] >= 0.7:
        return (
            "matched_q_feedback_causal",
            "Localize which matched nodes/actions contribute most, then test a conservative search-time Q-confidence mechanism.",
        )
    if primary["all_qsync_rest"]["rescue_rate"]["estimate"] < 0.3:
        return (
            "matched_q_not_sufficient",
            "Instrument matched-tree coverage more deeply before changing ML/search behavior.",
        )
    return (
        "distributed_matched_q_feedback",
        "Localize which matched nodes/actions contribute most, then test a conservative search-time Q-confidence mechanism.",
    )


def _serialized(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in record.items() if key != "lanes"},
        "lanes": {
            name: {key: value for key, value in lane.items() if key != "summary"}
            for name, lane in record["lanes"].items()
        },
    }


def _report(summary: dict[str, Any]) -> str:
    lanes = ("root_qsync", "all_qsync_32", "all_qsync_rest")
    primary_rows = [
        "| Root hash | root_qsync | all_qsync_32 | all_qsync_rest |",
        "| --- | --- | --- | --- |",
    ]
    for record in summary["primary_records"]:
        primary_rows.append(
            "| `{}` | {} | {} | {} |".format(
                record["state_hash"],
                *(
                    "rescued"
                    if record["lanes"][lane]["final_move"] == record["p1_final_move"]
                    else "retained"
                    for lane in lanes
                ),
            )
        )
    control_rows = [
        "| Root hash | full | root_qsync | all_qsync_32 | all_qsync_rest |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in summary["records"]:
        if record["state_hash"] not in summary["negative_control_roots"]:
            continue
        control_rows.append(
            "| `{}` | {} | {} | {} | {} |".format(
                record["state_hash"],
                *(
                    "new divergence"
                    if record["lanes"][lane]["final_move"] != record["p1_final_move"]
                    else "stable"
                    for lane in ("full", *lanes)
                ),
            )
        )
    return "\n".join(
        [
            "# Matched Selection-Q Feedback",
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
            "## Frozen 40-Root Rescue Table",
            "",
            *primary_rows,
            "",
            "## Negative Controls",
            "",
            "```json",
            json.dumps(summary["negative_controls"], indent=2, sort_keys=True),
            "```",
            "",
            *control_rows,
            "",
            "## Coverage Telemetry",
            "",
            "```json",
            json.dumps(summary["coverage_telemetry"], indent=2, sort_keys=True),
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
        / "docs/data/alphazero-lite-fresh-p1-adapter-matched-q-feedback-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-matched-q-feedback-results.md",
    )
    args = parser.parse_args()
    frozen, prior, manifest = (
        json.loads(path.read_text()) for path in (FROZEN, PR222, MANIFEST)
    )
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    manifest_by_hash = {row["state_hash"]: row for row in manifest["rows"]}
    frozen_hashes = frozen["full_amplified_1200"]
    controls_metadata = _control_subset(
        prior["records"], [prior_by_hash[key] for key in frozen_hashes]
    )
    selected = [manifest_by_hash[key] for key in frozen_hashes] + [
        manifest_by_hash[row["state_hash"]] for row in controls_metadata
    ]
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    tasks = [
        (
            decode_kalah_v3_base_state(list(rows[int(meta["replay_index"])]["state"])),
            meta,
        )
        for meta in selected
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
    controls = [by_hash[row["state_hash"]] for row in controls_metadata]
    primary = {
        lane: {
            "rescue_rate": _bootstrap(
                [
                    row["lanes"][lane]["final_move"] == row["p1_final_move"]
                    for row in primary_records
                ]
            ),
            "retention_rate": _bootstrap(
                [
                    row["lanes"][lane]["final_move"] != row["p1_final_move"]
                    for row in primary_records
                ]
            ),
            "rescued_root_hashes": [
                row["state_hash"]
                for row in primary_records
                if row["lanes"][lane]["final_move"] == row["p1_final_move"]
            ],
            "retained_root_hashes": [
                row["state_hash"]
                for row in primary_records
                if row["lanes"][lane]["final_move"] != row["p1_final_move"]
            ],
        }
        for lane in LANES
        if lane != "full"
    }
    negative = {
        lane: sum(
            row["lanes"][lane]["final_move"] != row["p1_final_move"] for row in controls
        )
        for lane in LANES
    }
    negative["new_divergences_all_qsync_rest"] = (
        negative["all_qsync_rest"] - negative["full"]
    )
    _init_worker(str(p1_artifact), str(a16_artifact))
    held_out = [
        _audit((row["state"], {"state_hash": row["state_hash"]}))
        for row in json.loads(PR220.read_text())["first_game_divergences"]
    ]
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
        "all_selection_q_lane_invariants": all(
            all(row["lanes"][lane]["invariants"].values())
            for row in records
            for lane in LANES
            if lane != "full"
        ),
    }
    classification, follow_up = _classification(invariants, primary, negative)
    coverage = {
        "reference_capture": dict(
            sum(
                (Counter(row["reference_capture_telemetry"]) for row in records),
                Counter(),
            )
        ),
        **{
            lane: dict(
                sum(
                    (
                        Counter(row["lanes"][lane]["coverage_telemetry"])
                        for row in records
                    ),
                    Counter(),
                )
            )
            for lane in LANES
            if lane != "full"
        },
    }
    summary = {
        "schema": "azlite_matched_q_feedback_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "priors_modified": False,
            "backups_modified": False,
            "selection_q_only": True,
            "c_puct": C_PUCT,
            "fpu_mode": "zero",
            "simulations": SIMULATIONS,
            "root_noise": False,
            "intervention_start": "d+1",
        },
        "hashes": {
            "frozen_identity": frozen["identity_hash"],
            "replay": sha256_file(replay),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
            "reference_cache": hashlib.sha256(
                json.dumps(
                    [row["reference_cache_sha"] for row in records],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "invariants": invariants,
        "frozen_amplified_roots": frozen_hashes,
        "negative_control_roots": [row["state_hash"] for row in controls_metadata],
        "lanes": list(LANES),
        "primary_outcome": primary,
        "negative_controls": negative,
        "coverage_telemetry": coverage,
        "records": [_serialized(row) for row in records],
        "primary_records": [_serialized(row) for row in primary_records],
        "held_out_pr220": [_serialized(row) for row in held_out],
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    for path in (args.out_summary, args.out_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
