#!/usr/bin/env python3
# ruff: noqa: E402
"""Preregistered root-only self-Q confidence screen for the PR #226 roots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.postdivergence_amplification import visit_js
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    _bootstrap,
    _game_hash,
)
from ml.alphazero_lite.run_fresh_p1_adapter_q_feedback_necessity import (
    C_PUCT,
    SIMULATIONS,
    _seed,
    decode_kalah_v3_base_state,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_synchronized_backup import _control_subset
from ml.alphazero_lite.self_play import PUCT, Node, root_q_confidence_override

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
ALPHAS = (1.0, 0.75, 0.50, 0.25, 0.00)
CHECKPOINTS = (1, 4, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1200)
BOOTSTRAP_SAMPLES = 10_000
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def _lane_name(alpha: float) -> str:
    return f"alpha_{int(alpha * 100):03d}"


def _init_worker(p1_artifact: str, a16_artifact: str) -> None:
    global _P1, _A16
    _P1 = ArtifactEvaluator(Path(p1_artifact))
    _A16 = ArtifactEvaluator(Path(a16_artifact))


def _search(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    seed: int,
    *,
    override: Any = None,
    hook: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        selection_q_override=override,
        pre_simulation_hook=hook,
        root_snapshot_checkpoints=set(CHECKPOINTS),
        record_root_trajectory=True,
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    summary = search.root_summary()
    return summary, summary["root_trajectory"]


def _root_reference_hook(references: dict[int, dict[int, float]]) -> Any:
    def hook(simulation: int, root: Node) -> None:
        references[simulation] = {
            int(move): float(child.q_value)
            for move, child in root.children.items()
            if child.visit_count > 0
        }

    return hook


def _root_qsync(
    root_hash: str, references: dict[int, dict[int, float]], start: int
) -> Any:
    def override(
        simulation: int, state_hash: str, move: int, _q: float, visits: int
    ) -> float | None:
        if simulation < start or state_hash != root_hash or visits <= 0:
            return None
        return references.get(simulation, {}).get(move)

    return override


def _snapshot_by_checkpoint(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["simulation"]): row for row in summary["root_snapshots"]}


def _mechanism(
    full: dict[str, Any],
    lane: dict[str, Any],
    p1: dict[str, Any],
    references: dict[int, dict[int, float]],
) -> list[dict[str, Any]]:
    full_by_t, lane_by_t, p1_by_t = map(_snapshot_by_checkpoint, (full, lane, p1))
    rows = []
    for checkpoint in CHECKPOINTS:
        baseline, observed, reference = (
            full_by_t[checkpoint],
            lane_by_t[checkpoint],
            p1_by_t[checkpoint],
        )
        raw = {
            int(row["move"]): float(row["stored_q_value"]) for row in observed["moves"]
        }
        selected = {
            int(row["move"]): float(row["selection_q_value"])
            for row in observed["moves"]
        }
        p1_q = {
            int(row["move"]): float(references[checkpoint].get(int(row["move"]), 0.0))
            for row in reference["moves"]
        }
        scores = sorted(
            observed["moves"], key=lambda row: (-row["selection_score"], row["move"])
        )
        actions = sorted(set(raw) | set(p1_q))
        rows.append(
            {
                "simulation": checkpoint,
                "raw_stored_q": raw,
                "selection_q": selected,
                "raw_q_spread": max(raw.values()) - min(raw.values()),
                "selection_q_spread": max(selected.values()) - min(selected.values()),
                "u": {
                    int(row["move"]): float(row["u_component"])
                    for row in observed["moves"]
                },
                "q_plus_u_margin": float(
                    scores[0]["selection_score"] - scores[1]["selection_score"]
                )
                if len(scores) > 1
                else None,
                "visit_distribution": observed["visits"],
                "visit_js_vs_full_p1": visit_js(
                    np.asarray(observed["visits"]), np.asarray(reference["visits"])
                ),
                "visit_leader": max(
                    observed["moves"],
                    key=lambda row: (row["visit_count"], -row["move"]),
                )["move"],
                "deterministic_root_move": observed["selected_move"],
                "versus_root_qsync": None,
                "versus_p1_reference_q_used_by_root_qsync": {
                    "q_l1": sum(
                        abs(selected.get(a, 0.0) - p1_q.get(a, 0.0)) for a in actions
                    ),
                    "max_abs_q_difference": max(
                        abs(selected.get(a, 0.0) - p1_q.get(a, 0.0)) for a in actions
                    ),
                    "ranking_agreement": sorted(raw, key=lambda a: (-selected[a], a))
                    == sorted(p1_q, key=lambda a: (-p1_q[a], a)),
                    "best_q_action_agreement": max(
                        selected, key=lambda a: (selected[a], -a)
                    )
                    == max(p1_q, key=lambda a: (p1_q[a], -a)),
                    "p1_q_spread": max(p1_q.values()) - min(p1_q.values()),
                },
                "full_a16_raw_q": {
                    int(row["move"]): float(row["stored_q_value"])
                    for row in baseline["moves"]
                },
            }
        )
    return rows


def _divergence(
    reference: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> dict[str, Any]:
    differing = [
        (left, right)
        for left, right in zip(reference, observed, strict=True)
        if left["visit_leader"] != right["visit_leader"]
        or left["deterministic_move"] != right["deterministic_move"]
    ]
    return {
        "first_visit_leader_divergence": next(
            (
                left["simulation"]
                for left, right in differing
                if left["visit_leader"] != right["visit_leader"]
            ),
            None,
        ),
        "first_root_move_divergence": next(
            (
                left["simulation"]
                for left, right in differing
                if left["deterministic_move"] != right["deterministic_move"]
            ),
            None,
        ),
        "later_reconverges": bool(differing)
        and reference[-1]["deterministic_move"] == observed[-1]["deterministic_move"],
    }


def _audit(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    state, metadata = task
    assert _P1 is not None and _A16 is not None
    seed, root_hash = (
        _seed(metadata["state_hash"]),
        _game_hash(KalahGame.from_state(state)),
    )
    full, full_trajectory = _search(_A16, state, seed)
    references: dict[int, dict[int, float]] = {}
    p1, p1_trajectory = _search(_P1, state, seed, hook=_root_reference_hook(references))
    lanes: dict[str, Any] = {}
    for alpha in ALPHAS:
        name = _lane_name(alpha)
        summary, trajectory = _search(
            _A16, state, seed, override=root_q_confidence_override(root_hash, alpha)
        )
        lanes[name] = {
            "alpha": alpha,
            "final_move": summary["selected_move"],
            "mechanism": _mechanism(full, summary, p1, references),
            "drift": _divergence(full_trajectory, trajectory),
        }
    qsync_start = (
        int(
            metadata.get(
                "actual_first_divergence", metadata.get("first_divergence", {})
            ).get("simulation", 0)
        )
        + 1
    )
    qsync, qsync_trajectory = _search(
        _A16,
        state,
        seed,
        override=_root_qsync(root_hash, references, start=qsync_start),
    )
    lanes["root_qsync"] = {
        "final_move": qsync["selected_move"],
        "mechanism": _mechanism(full, qsync, p1, references),
        "drift": _divergence(full_trajectory, qsync_trajectory),
    }
    qsync_by_checkpoint = {
        row["simulation"]: row for row in lanes["root_qsync"]["mechanism"]
    }
    for alpha in ALPHAS:
        for row in lanes[_lane_name(alpha)]["mechanism"]:
            qsync_row = qsync_by_checkpoint[row["simulation"]]
            row["root_qsync_selection_q"] = qsync_row["selection_q"]
            row["versus_root_qsync"] = {
                "selection_q_l1": sum(
                    abs(
                        row["selection_q"].get(action, 0.0)
                        - qsync_row["selection_q"].get(action, 0.0)
                    )
                    for action in set(row["selection_q"])
                    | set(qsync_row["selection_q"])
                )
            }
    return {
        **metadata,
        "search_seed": seed,
        "p1_final_move": p1["selected_move"],
        "full_a16_final_move": full["selected_move"],
        "lanes": lanes,
        "root_hash": root_hash,
        "root_qsync_start": qsync_start,
    }


def _selection(primary: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    passing = [
        alpha
        for alpha in ALPHAS
        if primary[_lane_name(alpha)]["rescue_rate"]["estimate"] >= 0.70
        and controls[_lane_name(alpha)]["new_divergence_rate"]["estimate"] <= 0.10
    ]
    return {
        "rule": "largest alpha with rescue_rate >= 0.70 and new_divergence_rate <= 0.10",
        "preregistered_alphas": list(ALPHAS),
        "selected_alpha": max(passing) if passing else None,
        "passing_alphas": passing,
    }


def _report(summary: dict[str, Any]) -> str:
    rows = ["| Lane | Rescue | New divergence |", "| --- | ---: | ---: |"]
    for lane in (*(_lane_name(alpha) for alpha in ALPHAS), "root_qsync"):
        rescue = summary["frozen_screen"][lane]["rescue_rate"]
        control = summary["negative_controls"][lane]["new_divergence_rate"]
        rows.append(
            f"| `{lane}` | {rescue['estimate']:.3f} [{rescue['lower_95']:.3f}, {rescue['upper_95']:.3f}] | "
            f"{control['estimate']:.3f} [{control['lower_95']:.3f}, {control['upper_95']:.3f}] |"
        )
    lanes = [*(_lane_name(alpha) for alpha in ALPHAS), "root_qsync"]
    by_hash = {row["state_hash"]: row for row in summary["records"]}
    frozen_table = [
        "| Root hash | " + " | ".join(lanes) + " |",
        "| --- | " + " | ".join("---" for _ in lanes) + " |",
    ]
    for root_hash in summary["frozen_amplified_roots"]:
        record = by_hash[root_hash]
        frozen_table.append(
            "| `{}` | {} |".format(
                root_hash,
                " | ".join(
                    "rescued"
                    if record["lanes"][lane]["final_move"] == record["p1_final_move"]
                    else "retained"
                    for lane in lanes
                ),
            )
        )
    control_table = [
        "| Root hash | " + " | ".join(lanes) + " |",
        "| --- | " + " | ".join("---" for _ in lanes) + " |",
    ]
    for root_hash in summary["negative_control_roots"]:
        record = by_hash[root_hash]
        control_table.append(
            "| `{}` | {} |".format(
                root_hash,
                " | ".join(
                    "new divergence"
                    if record["lanes"][lane]["final_move"] != record["p1_final_move"]
                    else "stable"
                    for lane in lanes
                ),
            )
        )
    held_out_table = [
        "| Root hash | P1 | Full A16 | " + " | ".join(lanes) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in lanes) + " |",
    ]
    for record in summary["held_out_pr220"]:
        held_out_table.append(
            "| `{}` | {} | {} | {} |".format(
                record["state_hash"],
                record["p1_final_move"],
                record["full_a16_final_move"],
                " | ".join(str(record["lanes"][lane]["final_move"]) for lane in lanes),
            )
        )
    return "\n".join(
        [
            "# Root-Q Confidence Screen",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended next experiment:** {summary['recommended_next_experiment']}",
            "",
            "## Frozen Screen",
            "",
            *rows,
            "",
            "## Alpha Selection",
            "",
            "```json",
            json.dumps(summary["alpha_selection"], indent=2, sort_keys=True),
            "```",
            "",
            "## Frozen Rescue Table",
            "",
            *frozen_table,
            "",
            f"Persistent PR #225/#226 hard root: `{summary['persistent_hard_root']}`.",
            "",
            "## Washed-Out Controls",
            "",
            *control_table,
            "",
            "## Held-Out PR #220 Roots",
            "",
            *held_out_table,
            "",
            "## Mechanism And Gates",
            "",
            "Per-root checkpoint telemetry and root_qsync Q comparisons are in the JSON summary under `records[].lanes[].mechanism`.",
            "Symmetric arena and parent-search drift were not run because `selected_alpha` is null.",
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
        / "docs/data/alphazero-lite-fresh-p1-adapter-root-q-confidence-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-root-q-confidence-results.md",
    )
    args = parser.parse_args()
    frozen, prior, manifest = (
        json.loads(path.read_text()) for path in (FROZEN, PR222, MANIFEST)
    )
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    manifest_by_hash = {row["state_hash"]: row for row in manifest["rows"]}
    amplified = frozen["full_amplified_1200"]
    controls_meta = _control_subset(
        prior["records"], [prior_by_hash[key] for key in amplified]
    )
    selected = [manifest_by_hash[key] for key in amplified] + [
        manifest_by_hash[row["state_hash"]] for row in controls_meta
    ]
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    replay_rows = read_jsonl(replay)
    tasks = [
        (
            decode_kalah_v3_base_state(
                list(replay_rows[int(meta["replay_index"])]["state"])
            ),
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
    primary_records, control_records = (
        [by_hash[key] for key in amplified],
        [by_hash[row["state_hash"]] for row in controls_meta],
    )
    lanes = [*(_lane_name(alpha) for alpha in ALPHAS), "root_qsync"]
    screen = {
        lane: {
            "rescue_rate": _bootstrap(
                [
                    row["lanes"][lane]["final_move"] == row["p1_final_move"]
                    for row in primary_records
                ]
            ),
            "rescued_hashes": [
                row["state_hash"]
                for row in primary_records
                if row["lanes"][lane]["final_move"] == row["p1_final_move"]
            ],
            "retained_hashes": [
                row["state_hash"]
                for row in primary_records
                if row["lanes"][lane]["final_move"] != row["p1_final_move"]
            ],
        }
        for lane in lanes
    }
    controls = {
        lane: {
            "new_divergence_rate": _bootstrap(
                [
                    row["lanes"][lane]["final_move"] != row["p1_final_move"]
                    for row in control_records
                ]
            ),
            "divergent_hashes": [
                row["state_hash"]
                for row in control_records
                if row["lanes"][lane]["final_move"] != row["p1_final_move"]
            ],
        }
        for lane in lanes
    }
    alpha_selection = _selection(screen, controls)
    invariants = {
        "full_a16_amplified_reproduces": all(
            row["full_a16_final_move"] != row["p1_final_move"]
            for row in primary_records
        ),
        "alpha_100_identity": all(
            row["lanes"]["alpha_100"]["final_move"] == row["full_a16_final_move"]
            for row in records
        ),
        "root_qsync_reproduces_39_of_40": len(screen["root_qsync"]["rescued_hashes"])
        == 39,
        "root_qsync_controls_stable": not controls["root_qsync"]["divergent_hashes"],
    }
    if not all(invariants.values()):
        classification = "invariant_failure"
    elif alpha_selection["selected_alpha"] is None:
        classification = "exact_parent_q_direction_required"
    else:
        classification = "inconclusive"
    _init_worker(str(p1_artifact), str(a16_artifact))
    held_out_rows = {
        row["state_hash"]: row
        for row in json.loads(
            (
                REPO_ROOT
                / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
            ).read_text()
        )["first_game_divergences"]
    }
    held_out = [
        _audit((row["state"], {"state_hash": row["state_hash"], **row}))
        for row in held_out_rows.values()
    ]
    summary = {
        "schema": "azlite_root_q_confidence_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "p1_q_in_alpha_lanes": False,
            "selection_q_only": True,
            "simulations": SIMULATIONS,
            "c_puct": C_PUCT,
            "fpu_mode": "zero",
            "normalize_values": False,
            "root_noise": False,
        },
        "config": {"alphas": list(ALPHAS), "checkpoints": list(CHECKPOINTS)},
        "hashes": {
            "frozen_identity": frozen["identity_hash"],
            "replay": sha256_file(replay),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
        },
        "frozen_amplified_roots": amplified,
        "negative_control_roots": [row["state_hash"] for row in controls_meta],
        "frozen_screen": screen,
        "negative_controls": controls,
        "alpha_selection": alpha_selection,
        "invariants": invariants,
        "persistent_hard_root": "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
        "records": records,
        "held_out_pr220": held_out,
        "classification": classification,
        "recommended_next_experiment": "Audit candidate-vs-parent root-Q convergence/error as a function of child visits and estimate Q confidence from search statistics.",
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
