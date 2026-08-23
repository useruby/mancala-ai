#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate PR #214 A16 with exact lagged-P1 root-Q shadow searches only."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, run_arena_worker, sha256_file
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import read_jsonl
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    FROZEN,
    MANIFEST,
    PR222,
    _control_subset,
    _seed,
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import ARENA_SUITE
from ml.alphazero_lite.shadow_root_q import run_shadow_root_q_search
from ml.alphazero_lite.self_play import PUCT

P0 = REPO_ROOT / "model-artifact/current"
P0_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
P1_SHA = "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12"
A16_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
CONTEXTS = ("384:256", "1200:1200")


def ordinary(
    game: KalahGame, evaluator: ArtifactEvaluator, simulations: int, seed: int
) -> dict:
    search = PUCT(
        evaluator,
        simulations,
        1.25,
        random.Random(seed),
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, _root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return {"visits": visits.tolist(), "summary": search.root_summary()}


def frozen_contract(
    p1: ArtifactEvaluator, a16: ArtifactEvaluator, replay: Path
) -> dict[str, Any]:
    frozen, prior, manifest = (
        json.loads(path.read_text()) for path in (FROZEN, PR222, MANIFEST)
    )
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    manifest_by_hash = {row["state_hash"]: row for row in manifest["rows"]}
    primary_hashes = frozen["full_amplified_1200"]
    controls = _control_subset(
        prior["records"], [prior_by_hash[key] for key in primary_hashes]
    )
    rows = read_jsonl(replay)
    records = []
    for meta in [manifest_by_hash[key] for key in primary_hashes] + [
        manifest_by_hash[row["state_hash"]] for row in controls
    ]:
        game = KalahGame.from_state(
            decode_kalah_v3_base_state(list(rows[int(meta["replay_index"])]["state"]))
        )
        seed = _seed(meta["state_hash"])
        parent = ordinary(game, p1, 1200, seed)
        base = ordinary(game, a16, 1200, seed)
        _visits, _root, shadow = run_shadow_root_q_search(
            game,
            main_evaluator=a16,
            shadow_evaluator=p1,
            simulations=1200,
            c_puct=1.25,
            seed=seed,
            root_policy_mode="deterministic",
        )
        self_base = ordinary(game, p1, 1200, seed)
        self_visits, self_root, self_shadow = run_shadow_root_q_search(
            game,
            main_evaluator=p1,
            shadow_evaluator=p1,
            simulations=1200,
            c_puct=1.25,
            seed=seed,
            root_policy_mode="deterministic",
        )
        self_summary = self_shadow["main_summary"]
        self_identity = (
            self_base["summary"]["selected_move"] == self_summary["selected_move"]
            and self_base["visits"] == self_visits.tolist()
            and self_base["summary"]["child_stats"] == self_summary["child_stats"]
            and self_base["summary"]["root_q_value"] == self_summary["root_q_value"]
        )
        records.append(
            {
                "state_hash": meta["state_hash"],
                "primary": meta["state_hash"] in primary_hashes,
                "p1_move": parent["summary"]["selected_move"],
                "a16_move": base["summary"]["selected_move"],
                "shadow_move": shadow["main_summary"]["selected_move"],
                "self_identity": self_identity,
                "no_future_information": shadow["no_future_information"],
            }
        )
    primary = [r for r in records if r["primary"]]
    control = [r for r in records if not r["primary"]]
    return {
        "records": records,
        "rescue": sum(r["shadow_move"] == r["p1_move"] for r in primary) / len(primary),
        "new_divergences": sum(
            r["shadow_move"] != r["p1_move"] and r["a16_move"] == r["p1_move"]
            for r in control
        ),
        "self_shadow_identity": all(r["self_identity"] for r in records),
        "no_future_information": all(r["no_future_information"] for r in records),
    }


def arena_records(
    workdir: Path,
    challenger: Path,
    current: Path,
    shadow: Path | None,
    context: str,
    role: str,
    workers: int,
) -> list[dict]:
    challenger_sims, current_sims = map(int, context.split(":"))
    records: list[dict] = []
    for seat in (0, 1):
        counts = [256 // workers + (index < 256 % workers) for index in range(workers)]
        starts, cursor = [], 0
        for count in counts:
            starts.append(cursor)
            cursor += count
        kwargs = {
            "challenger_path": str(challenger),
            "current_path": str(current),
            "challenger_shadow_artifact": None if shadow is None else str(shadow),
            "challenger_simulations": challenger_sims,
            "current_simulations": current_sims,
            "seed": 42,
            "c_puct": 1.25,
            "max_moves": 200,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
            "games_per_opening": 2,
            "challenger_starts": seat,
            "opening_prefixes_jsonl": str(ARENA_SUITE),
            "suite_sha256_override": sha256_file(ARENA_SUITE),
        }
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_arena_worker,
                    worker_id=index,
                    start_index=start,
                    games=count,
                    **kwargs,
                )
                for index, (start, count) in enumerate(zip(starts, counts, strict=True))
                if count
            ]
            results = [future.result() for future in futures]
        telemetry = []
        for result in results:
            for row in result["game_entries"]:
                row["opponent_weights_sha256"] = sha256_file(current / "weights.json")
                row["opponent_config_sha256"] = f"{context}:1.25"
            records.extend(result["game_entries"])
            telemetry.extend(result["shadow_move_telemetry"])
        (workdir / context / role / f"seat_{seat}.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (workdir / context / role / f"seat_{seat}.json").write_text(
            json.dumps(telemetry, indent=2) + "\n"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_lagged_parent_shadow_q")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-lagged-parent-shadow-q-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-lagged-parent-shadow-q-results.md",
    )
    parser.add_argument("--frozen-only", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    p1_path = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_path = args.adapter_workdir / "artifacts/step_0016/artifact"
    hashes = {
        "p0": sha256_file(P0 / "weights.json"),
        "p1": sha256_file(p1_path / "weights.json"),
        "a16": sha256_file(a16_path / "weights.json"),
        "canonical_suite": sha256_file(ARENA_SUITE),
    }
    p1, a16 = ArtifactEvaluator(p1_path), ArtifactEvaluator(a16_path)
    frozen = frozen_contract(p1, a16, args.adapter_workdir / "fresh_p1_self_play.jsonl")
    invariants = {
        "artifact_hashes": hashes["p0"] == P0_SHA
        and hashes["p1"] == P1_SHA
        and hashes["a16"] == A16_SHA,
        "frozen_rescue": frozen["rescue"] >= 0.95 and frozen["new_divergences"] == 0,
        "self_shadow_identity": frozen["self_shadow_identity"],
        "no_future_information": frozen["no_future_information"],
    }
    contexts: dict[str, Any] = {}
    if all(invariants.values()) and not args.frozen_only:
        for context in CONTEXTS:
            control = arena_records(
                args.workdir,
                p1_path,
                p1_path,
                p1_path,
                context,
                "p1_self_shadow",
                args.workers,
            )
            treatment = arena_records(
                args.workdir,
                a16_path,
                p1_path,
                p1_path,
                context,
                "a16_p1_shadow",
                args.workers,
            )
            effect = paired_opening_candidate_effect(treatment, control)
            contexts[context] = {
                "raw_score": effect["orientation_decomposition"][
                    "candidate_challenger_score"
                ],
                "control_score": effect["orientation_decomposition"][
                    "current_control_challenger_score"
                ],
                "paired_treatment_effect": effect["paired_candidate_effect"],
                "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                "seat_a_effect": effect["p0_effect"],
                "seat_b_effect": effect["p1_effect"],
                "win_draw_loss": {
                    "wins": sum(r["winner"] == "challenger" for r in treatment),
                    "draws": sum(r["winner"] == "draw" for r in treatment),
                    "losses": sum(r["winner"] == "current" for r in treatment),
                },
            }
    classification = (
        "invariant_failure" if not all(invariants.values()) else "inconclusive"
    )
    if (
        not args.frozen_only
        and all(invariants.values())
        and all(
            contexts[key]["opening_bootstrap_ci"]["lower_95"] >= 0 for key in CONTEXTS
        )
    ):
        classification = "lagged_parent_shadow_rescues_game_strength"
    elif (
        not args.frozen_only
        and all(invariants.values())
        and contexts.get("384:256", {})
        .get("opening_bootstrap_ci", {})
        .get("lower_95", -1)
        >= 0
    ):
        classification = "shadow_q_rescues_low_budget_only"
    elif not args.frozen_only and all(invariants.values()) and frozen["rescue"] >= 0.95:
        classification = "frozen_root_rescue_does_not_transfer"
    summary = {
        "schema": "azlite_lagged_parent_shadow_q_v1",
        "hashes": hashes,
        "guardrails": {
            "training": False,
            "self_play": False,
            "weights_modified": False,
            "main_and_shadow_simulations_equal": True,
            "c_puct": 1.25,
        },
        "invariants": invariants,
        "frozen": frozen,
        "contexts": contexts,
        "compute_accounting": {
            "main_simulations_per_move": "context challenger budget",
            "shadow_simulations_per_move": "same as main",
            "multiplier_vs_ordinary": 2.0,
        },
        "classification": classification,
        "recommended_follow_up": "Run one fresh AlphaZero generation where candidate evaluation uses its parent as a lagged shadow-Q search, while training remains unchanged.",
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(
        "# Lagged-Parent Shadow Root-Q\n\n**Classification:** `{}`\n\n```json\n{}\n```\n".format(
            classification, json.dumps(summary, indent=2, sort_keys=True)
        )
    )
    print(classification)


if __name__ == "__main__":
    main()
