#!/usr/bin/env python3
"""Test whether PR #204's high legal-policy-L1 tail causes its MCTS regression.

This is diagnostic-only: P1/P2 are frozen artifacts and the intervention uses
the existing all-depth PUCT prior callback.  It never trains or self-plays.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    ArtifactEvaluator,
    evaluate_artifact_position,
    run_arena_worker,
)
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import (  # noqa: E402
    decode_state,
)
from ml.alphazero_lite.hard_projection import TrustStateSet  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.policy_prior_localization import (  # noqa: E402
    TailPriorSubstitutionOverride,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_gen2_hard_trust_region import (  # noqa: E402
    PR204_GEN2_REPLAY_HASH,
    PR204_UNPROJECTED_S46_STATE_HASH,
    PR206_TRUST_SET_HASH,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P1_EXPECTED_STATE_HASH,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _new_model  # noqa: E402
from ml.alphazero_lite.run_policy_detached_trunk_ablation import ARENA_SUITE  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import stable_hash  # noqa: E402
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import load_checkpoint_into_model  # noqa: E402

NAMESPACE = "azlite_gen2_tail_prior_override_v1"
PERCENTILES = (50, 75, 90, 95, 99)
LANES = (
    "candidate_all",
    "tail_q99",
    "tail_q95",
    "tail_q90",
    "tail_q75",
    "incumbent_all",
)


def _state_hash(path: Path) -> str:
    model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(model, path)
    return stable_hash(
        {
            k: v.detach().cpu().numpy().tobytes().hex()
            for k, v in model.state_dict().items()
        }
    )


def _legal_l1(
    candidate: ArtifactEvaluator, parent: ArtifactEvaluator, row: dict[str, Any]
) -> float:
    game = KalahGame.from_state(decode_state(row["state"]))
    legal = game.possible_moves()
    cp, _ = candidate.evaluate(game)
    pp, _ = parent.evaluate(game)
    c = np.zeros(6, dtype=np.float64)
    p = np.zeros(6, dtype=np.float64)
    c[legal] = cp[legal]
    p[legal] = pp[legal]
    c[legal] /= c[legal].sum()
    p[legal] /= p[legal].sum()
    return float(np.abs(c - p).sum())


def calibrate(
    rows: list[dict[str, Any]], candidate: ArtifactEvaluator, parent: ArtifactEvaluator
) -> dict[str, Any]:
    """Freeze thresholds exclusively from unique Gen-2 replay states."""
    seen: set[tuple[float, ...]] = set()
    unique = []
    for row in rows:
        key = tuple(float(v) for v in row["state"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    values = np.asarray(
        [_legal_l1(candidate, parent, row) for row in unique], dtype=np.float64
    )
    return {
        "unique_state_count": len(unique),
        "thresholds": {f"q{q}": float(np.percentile(values, q)) for q in PERCENTILES},
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _telemetry(log: list[dict[str, Any]]) -> dict[str, Any]:
    l1 = np.asarray(
        [
            entry.get(
                "candidate_vs_parent_legal_l1", entry.get("pairwise_legal_l1", 0.0)
            )
            for entry in log
        ],
        dtype=np.float64,
    )
    overridden = np.asarray([entry["substituted"] for entry in log], dtype=bool)
    by_depth: dict[str, dict[str, float | int]] = {}
    by_player: dict[str, dict[str, float | int]] = {}
    for key, field in ((by_depth, "depth"), (by_player, "player_to_move")):
        for value in sorted({int(entry.get(field, -1)) for entry in log}):
            select = np.asarray([int(entry.get(field, -1)) == value for entry in log])
            key[str(value)] = {
                "expanded_nodes": int(select.sum()),
                "overridden_nodes": int((overridden & select).sum()),
                "override_fraction": float(overridden[select].mean()),
            }
    return {
        "total_expanded_nodes": len(log),
        "overridden_node_count": int(overridden.sum()),
        "override_fraction": float(overridden.mean()) if len(log) else 0.0,
        "override_fraction_by_depth": by_depth,
        "override_fraction_by_player_to_move": by_player,
        "mean_l1_overridden": float(l1[overridden].mean()) if overridden.any() else 0.0,
        "mean_l1_non_overridden": float(l1[~overridden].mean())
        if (~overridden).any()
        else 0.0,
        "policy_l1_mass_removed": float(l1[overridden].sum()),
        "policy_l1_mass_removed_fraction": float(l1[overridden].sum() / l1.sum())
        if l1.sum()
        else 0.0,
        "encountered_l1_percentiles": {
            f"p{q}": float(np.percentile(l1, q)) for q in PERCENTILES
        },
    }


def probe_lane(
    rows: list[dict[str, Any]],
    candidate: ArtifactEvaluator,
    parent: ArtifactEvaluator,
    threshold: float | None,
) -> dict[str, Any]:
    """Deterministic all-depth search probe; separate from arena outcomes."""
    override = TailPriorSubstitutionOverride(parent, threshold=threshold)
    options = build_eval_search_options(tactical_root_bias=0.0)
    for row in rows[:256]:
        evaluate_artifact_position(
            evaluator=candidate,
            state=decode_state(row["state"]),
            simulations=384,
            seed=42,
            c_puct=1.25,
            search_options=options,
            prior_override=override,
        )
    return _telemetry(override.telemetry_log)


def arena_lane(
    *,
    candidate: Path,
    parent: Path,
    lane: str,
    threshold: float | None,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the fixed canonical 128-opening, seat-swapped arena for one lane."""
    jobs = []
    worker_count = max(1, workers)
    # 256 games per forced seat gives two deterministic games for each opening.
    for seat in (0, 1):
        for worker in range(worker_count):
            start = worker * 256 // worker_count
            end = (worker + 1) * 256 // worker_count
            if end <= start:
                continue
            jobs.append(
                {
                    "worker_id": seat * worker_count + worker,
                    "start_index": start,
                    "games": end - start,
                    "challenger_path": str(candidate),
                    "current_path": str(parent),
                    "challenger_simulations": 384,
                    "current_simulations": 256,
                    "seed": 42,
                    "c_puct": 1.25,
                    "max_moves": 200,
                    "opening_prefixes_jsonl": str(ARENA_SUITE),
                    "games_per_opening": 2,
                    "challenger_starts": seat,
                    "root_policy_mode": "deterministic",
                    "root_temperature": 0.0,
                    "normalize_values": False,
                    "tactical_root_bias": 0.0,
                    "challenger_prior_override_mode": "incumbent_all"
                    if lane == "incumbent_all"
                    else None,
                    "challenger_prior_tail_threshold": threshold
                    if lane.startswith("tail_")
                    else None,
                }
            )
    records: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count * 2) as pool:
        for result in pool.map(_run_worker, jobs):
            records.extend(result["game_entries"])
            seat = int(result["game_entries"][0]["challenger_player"])
            for entry in result.get("challenger_prior_override_telemetry", []):
                telemetry.append({**entry, "candidate_seat": seat})
    return records, telemetry


def _run_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    return run_arena_worker(**kwargs)


def _wdl(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "wins": sum(record["winner"] == "challenger" for record in records),
        "draws": sum(record["winner"] == "draw" for record in records),
        "losses": sum(record["winner"] == "current" for record in records),
    }


def arena_telemetry(log: list[dict[str, Any]]) -> dict[str, Any]:
    result = _telemetry(log)
    result["by_candidate_seat"] = {
        str(seat): _telemetry(
            [entry for entry in log if entry["candidate_seat"] == seat]
        )
        for seat in (0, 1)
    }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Gen-2 Tail Prior Override Results",
        "",
        f"**Primary classification:** `{summary['classification']['label']}`",
        "",
        f"**Next experiment:** {summary['classification']['next_experiment']}",
        "",
        "## Frozen Calibration",
        "",
        f"- Gen-2 trust-state SHA256: `{summary['inputs']['trust_state_set_hash']}`",
        f"- P1 state hash: `{summary['inputs']['p1_state_hash']}`",
        f"- P2 state hash: `{summary['inputs']['p2_state_hash']}`",
        "",
        "| Quantile | Legal-policy L1 threshold |",
        "| --- | ---: |",
    ]
    calibration = summary["inputs"]["calibration"]
    for name, value in calibration["thresholds"].items():
        lines.append(f"| {name} | {value:.10f} |")
    if not summary["arena"]:
        return "\n".join(lines + ["", "Canonical arena not run.", ""])
    lines.extend(
        [
            "",
            "## Canonical Arena (384:256, 128 openings, seat swapped)",
            "",
            "| Lane | Effect | 95% CI | P0 | P1 | W/D/L | Recovery | P0 Recovery |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for lane in LANES:
        entry = summary["arena"][lane]
        ci = entry["opening_bootstrap_ci"]
        recovery = entry.get("recovery_fraction")
        p0_recovery = entry.get("p0_recovery_fraction")
        lines.append(
            f"| {lane} | {entry['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {entry['p0_effect']:+.4f} | {entry['p1_effect']:+.4f} | {entry['win_draw_loss']['wins']}/{entry['win_draw_loss']['draws']}/{entry['win_draw_loss']['losses']} | {recovery if recovery is not None else 'n/a'} | {p0_recovery if p0_recovery is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Override Coverage (frozen replay-state PUCT probe)",
            "",
            "| Lane | Expanded | Overridden | Fraction | L1 mass removed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane, entry in summary["probe_telemetry"].items():
        lines.append(
            f"| {lane} | {entry['total_expanded_nodes']} | {entry['overridden_node_count']} | {entry['override_fraction']:.4f} | {entry['policy_l1_mass_removed_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "High-budget gate: not run. No selective lane simultaneously recovered at least 70% and overrode at most 10% of expanded nodes.",
        ]
    )
    return "\n".join(lines) + "\n"


def classify(summary: dict[str, Any]) -> dict[str, str]:
    sanity = summary["sanity"]
    if not all(sanity.values()):
        return {
            "label": "invariant_failure",
            "next_experiment": "repair the failed invariant before further experiments",
        }
    arena = summary.get("arena", {})
    if not arena:
        return {
            "label": "inconclusive",
            "next_experiment": "run the preregistered canonical arena lanes",
        }
    incumbent = arena["incumbent_all"]
    if abs(incumbent["paired_candidate_effect"]) >= 0.02:
        return {
            "label": "invariant_failure",
            "next_experiment": "repair incumbent_all equivalence before further experiments",
        }
    tight = [arena[lane] for lane in ("tail_q99", "tail_q95", "tail_q90")]
    if any(
        entry["recovery_fraction"] >= 0.70
        and entry["telemetry"]["override_fraction"] <= 0.10
        for entry in tight
    ):
        return {
            "label": "high_drift_tail_causal",
            "next_experiment": "train Gen-2 with a selective per-state output constraint protecting only this high-drift tail",
        }
    q75 = arena["tail_q75"]
    if q75["recovery_fraction"] >= 0.70:
        return {
            "label": "moderately_sparse_tail_causal",
            "next_experiment": "train Gen-2 with a broader selective per-state output constraint",
        }
    if q75["recovery_fraction"] < 0.70:
        return {
            "label": "distributed_state_drift",
            "next_experiment": "do not pursue simple percentile-tail constraints",
        }
    return {
        "label": "inconclusive",
        "next_experiment": "measure search sensitivity before another intervention",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--gen1-replay",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor/fresh_self_play.jsonl"),
    )
    parser.add_argument(
        "--gen2-replay",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl"),
    )
    parser.add_argument(
        "--p2",
        type=Path,
        default=Path(
            "/tmp/azlite_gen2_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
        ),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-gen2-tail-prior-override-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-gen2-tail-prior-override-results.md",
    )
    args = parser.parse_args()

    p1_artifact, p1_npz, _p1_weights, _p1_npz_hash, p1_hash = reconstruct_and_freeze_p1(
        args.current, args.p1_workdir, args.gen1_replay, 24
    )
    p2_artifact = args.p2.parent / "artifact"
    replay_hash = sha256_file(args.gen2_replay)
    p2_hash = _state_hash(args.p2)
    p1_model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1_model, p1_npz)
    p2_model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(p2_model, args.p2)
    rows = read_jsonl(args.gen2_replay)
    trust = TrustStateSet.from_replay_rows(rows, p1_model, torch.device("cpu"))
    candidate, parent = ArtifactEvaluator(p2_artifact), ArtifactEvaluator(p1_artifact)
    frozen_prefixes = (
        "input_layer.",
        "residual_layers.",
        "value_hidden_layer.",
        "value_head.",
    )
    frozen_stacks_identical = all(
        torch.equal(value.cpu(), p2_model.state_dict()[name].cpu())
        for name, value in p1_model.state_dict().items()
        if name.startswith(frozen_prefixes)
    )
    sanity = {
        "p1_hash_verified": p1_hash == P1_EXPECTED_STATE_HASH,
        "p2_hash_verified": p2_hash == PR204_UNPROJECTED_S46_STATE_HASH,
        "replay_hash_verified": replay_hash == PR204_GEN2_REPLAY_HASH,
        "trust_set_hash_verified": trust.state_set_hash == PR206_TRUST_SET_HASH,
        "trunk_and_value_stacks_bit_identical": frozen_stacks_identical,
    }
    if not all(sanity.values()):
        raise RuntimeError(f"refusing calibration after invariant failure: {sanity}")
    calibration = calibrate(rows, candidate, parent)
    thresholds = calibration["thresholds"]
    lane_thresholds = {
        "candidate_all": None,
        "tail_q99": thresholds["q99"],
        "tail_q95": thresholds["q95"],
        "tail_q90": thresholds["q90"],
        "tail_q75": thresholds["q75"],
        "incumbent_all": None,
    }
    probe = {
        lane: probe_lane(rows, candidate, parent, threshold)
        for lane, threshold in lane_thresholds.items()
        if lane != "candidate_all"
    }
    arena: dict[str, Any] = {}
    if args.arena:
        print("[arena] running fixed canonical tail lanes", flush=True)
        control, _unused = arena_lane(
            candidate=p1_artifact,
            parent=p1_artifact,
            lane="candidate_all",
            threshold=None,
            workers=args.arena_workers,
        )
        for lane in LANES:
            print(f"[arena] {lane}", flush=True)
            records, telemetry_log = arena_lane(
                candidate=p2_artifact,
                parent=p1_artifact,
                lane=lane,
                threshold=lane_thresholds[lane],
                workers=args.arena_workers,
            )
            metrics = paired_opening_candidate_effect(records, control)
            metrics["win_draw_loss"] = _wdl(records)
            metrics["telemetry"] = (
                arena_telemetry(telemetry_log) if telemetry_log else None
            )
            arena[lane] = metrics
        baseline = arena["candidate_all"]
        for lane, entry in arena.items():
            entry["recovery_fraction"] = (
                (entry["paired_candidate_effect"] - baseline["paired_candidate_effect"])
                / -baseline["paired_candidate_effect"]
                if baseline["paired_candidate_effect"]
                else None
            )
            entry["p0_recovery_fraction"] = (
                (entry["p0_effect"] - baseline["p0_effect"]) / -baseline["p0_effect"]
                if baseline["p0_effect"]
                else None
            )
        sanity["candidate_all_reproduces_pr204"] = (
            abs(arena["candidate_all"]["paired_candidate_effect"] + 0.095703125) < 1e-12
            and abs(arena["candidate_all"]["p0_effect"] + 0.19140625) < 1e-12
            and abs(arena["candidate_all"]["p1_effect"]) < 1e-12
        )
        sanity["incumbent_all_equivalent"] = (
            abs(arena["incumbent_all"]["paired_candidate_effect"]) < 0.02
        )
    summary: dict[str, Any] = {
        "schema": NAMESPACE,
        "inputs": {
            "p1_state_hash": p1_hash,
            "p2_state_hash": p2_hash,
            "gen2_replay_sha256": replay_hash,
            "trust_state_set_hash": trust.state_set_hash,
            "calibration": calibration,
            "lane_thresholds": lane_thresholds,
        },
        "sanity": sanity,
        "probe_telemetry": probe,
        "arena": arena,
    }
    summary["classification"] = classify(summary)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_report.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["inputs"], indent=2))


if __name__ == "__main__":
    main()
