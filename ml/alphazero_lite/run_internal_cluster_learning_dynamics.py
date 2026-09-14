#!/usr/bin/env python3
"""Frozen, evaluation-only longitudinal audit of the PR #305 state cluster.

The module deliberately keeps the cluster manifest out of every pipeline input.
It prepares a copy of the locked production recipe only to bind self-play to its
immediate parent checkpoint, which is the existing self-play CLI contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state  # noqa: E402
from ml.alphazero_lite.forensic_suite import canonical_state_key  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_internal_state_cluster_provenance_audit import (
    cluster_signature,
)  # noqa: E402
from ml.alphazero_lite.self_play import CheckpointEvaluator  # noqa: E402
from ml.alphazero_lite.legal_policy_metrics import (  # noqa: E402
    normalize_policy_over_legal_actions,
    policy_entropy,
    zeroed_legal_policy,
)

PR305 = ROOT / "docs/data/alphazero-lite-internal-state-cluster-provenance-audit.json"
LABELS = (
    ROOT / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit-labels.json"
)
RECIPE = (
    ROOT
    / "ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root.json"
)
MANIFEST_SCHEMA = "azlite_internal_cluster_frozen_eval_v1"
RESULT_SCHEMA = "azlite_internal_cluster_learning_dynamics_v1"
SEEDS = (61, 62, 63)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def state_sha(state: dict[str, Any]) -> str:
    return sha256_bytes(canonical_state_key(state).encode())


def active_stones(state: dict[str, Any]) -> int:
    return sum(state["player_pits"]) + sum(state["opponent_pits"])


def approximate_phase(state: dict[str, Any]) -> str:
    """A state-only phase proxy, fixed before evaluating any checkpoint."""
    stores = int(state["player_store"]) + int(state["opponent_store"])
    return "early" if stores <= 8 else "mid" if stores <= 24 else "late"


def label_record(state: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    key = canonical_state_key(state)
    label = labels.get(key)
    if label is None:
        raise ValueError(f"missing exact outcome label: {key}")
    return label


def entry(
    *,
    state: dict[str, Any],
    label: dict[str, Any],
    membership: str,
    provenance: str,
    degrading_actions: list[int] | None = None,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    return {
        "id": f"{membership}-{state_sha(state)[:16]}",
        "state": state,
        "state_sha256": state_sha(state),
        "membership": membership,
        "training_ineligible": True,
        "excluded_from_generated_replay": True,
        "provenance": provenance,
        "structural_signature": cluster_signature(state),
        "legal_actions": game.possible_moves(),
        "active_stones": active_stones(state),
        "approximate_phase": approximate_phase(state),
        "exact_outcome_value": float(label["exact_root_value"]),
        "exact_outcome_optimal_actions": list(label["exact_outcome_optimal_actions"]),
        "degrading_actions": sorted(degrading_actions or []),
    }


def _historically_good(
    evaluator_a: CheckpointEvaluator,
    evaluator_b: CheckpointEvaluator,
    state: dict[str, Any],
    optimal: list[int],
) -> bool:
    legal = KalahGame.from_state(state).possible_moves()
    for evaluator in (evaluator_a, evaluator_b):
        policy, _ = evaluator.evaluate(KalahGame.from_state(state))
        top = min(legal, key=lambda action: (-float(policy[action]), action))
        if top not in optimal:
            return False
    return True


def build_frozen_set(
    *,
    pr305: dict[str, Any],
    labels: dict[str, Any],
    control_evaluator: CheckpointEvaluator | None = None,
    uniform_evaluator: CheckpointEvaluator | None = None,
) -> dict[str, Any]:
    """Build the immutable state set using only inherited artifacts and rules."""
    selected = pr305["selected_cluster"]
    signature = selected["definition"]
    state_rows = {row["canonical_state"]: row for row in pr305["states"]}
    cluster_entries: dict[str, dict[str, Any]] = {}

    for key in selected["states"]:
        row = state_rows[key]
        state = row["state"]
        cluster_entries[key] = entry(
            state=state,
            label=label_record(state, labels),
            membership="cluster_anchor",
            provenance="PR305 selected canonical state",
            degrading_actions=row["selected_degrading_actions"],
        )
        for child in row["children"].values():
            child_state = json.loads(child["canonical_state"])
            cluster_entries.setdefault(
                child["canonical_state"],
                entry(
                    state=child_state,
                    label=label_record(child_state, labels),
                    membership="one_ply_child",
                    provenance="PR305 legal child",
                ),
            )

    # PR305 records only recovered parents.  It must not be supplemented by a
    # reverse move generator.
    for row in state_rows.values():
        for coverage in row["replay_coverage"]["control"]:
            for parent in coverage.get("parents", []):
                parent_state = parent.get("state")
                if parent_state is not None:
                    key = canonical_state_key(parent_state)
                    cluster_entries.setdefault(
                        key,
                        entry(
                            state=parent_state,
                            label=label_record(parent_state, labels),
                            membership="one_ply_parent",
                            provenance="PR305 recovered replay provenance",
                        ),
                    )

    for key, label in labels.items():
        state = json.loads(key)
        if cluster_signature(state) == signature:
            cluster_entries.setdefault(
                key,
                entry(
                    state=state,
                    label=label,
                    membership="structural_cluster",
                    provenance="PR303/PR305 exact outcome label",
                ),
            )

    anchors = [
        item
        for item in cluster_entries.values()
        if item["membership"] == "cluster_anchor"
    ]
    if len(anchors) != 1:
        raise ValueError("PR305 selected cluster must have exactly one anchor")
    anchor = anchors[0]
    controls: list[dict[str, Any]] = []
    if control_evaluator is not None and uniform_evaluator is not None:
        for key in sorted(labels):
            state = json.loads(key)
            if (
                key in cluster_entries
                or state["current_player"] != anchor["state"]["current_player"]
            ):
                continue
            if len(KalahGame.from_state(state).possible_moves()) != len(
                anchor["legal_actions"]
            ):
                continue
            if active_stones(state) != anchor["active_stones"]:
                continue
            if approximate_phase(state) != anchor["approximate_phase"]:
                continue
            label = labels[key]
            if _historically_good(
                control_evaluator,
                uniform_evaluator,
                state,
                label["exact_outcome_optimal_actions"],
            ):
                controls.append(
                    entry(
                        state=state,
                        label=label,
                        membership="matched_control",
                        provenance="exact label matched before longitudinal evaluation; seed45 control and uniform1200 raw top-1 optimal",
                    )
                )
            if len(controls) >= len(anchors):
                break
    if len(controls) != len(anchors):
        raise ValueError(
            "no deterministic historically-good matched controls available"
        )

    entries = sorted([*cluster_entries.values(), *controls], key=lambda row: row["id"])
    if len({row["id"] for row in entries}) != len(entries):
        raise ValueError("frozen-set IDs must be unique")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "training_injection": False,
        "exact_labels_used_for_training": False,
        "selected_cluster": json.loads(signature),
        "entries": entries,
    }
    payload["set_sha256"] = sha256_bytes(canonical_json(payload).encode())
    return payload


def policy_metrics(
    policy: list[float], *, entry_row: dict[str, Any], value: float
) -> dict[str, Any]:
    legal = entry_row["legal_actions"]
    legacy = zeroed_legal_policy(policy, legal)
    normalized = normalize_policy_over_legal_actions(policy, legal)
    top = min(legal, key=lambda action: (-normalized[action], action))
    optimal = set(entry_row["exact_outcome_optimal_actions"])
    legacy_metric = {
        "policy": legacy.tolist(),
        "legal_mass": float(legacy.sum()),
        "illegal_mass": float(1.0 - legacy.sum()),
        "optimal_mass": float(sum(legacy[action] for action in optimal)),
        "degrading_action_mass": float(
            sum(legacy[action] for action in entry_row["degrading_actions"])
        ),
        "legacy_zeroed_entropy": policy_entropy(legacy),
    }
    legal_normalized_metric = {
        "policy": normalized.tolist(),
        "optimal_mass": float(sum(normalized[action] for action in optimal)),
        "degrading_action_mass": float(
            sum(normalized[action] for action in entry_row["degrading_actions"])
        ),
        "legal_normalized_entropy": policy_entropy(normalized),
    }
    return {
        # Primary fields are canonical for all new frozen-set checkpoint evaluation.
        "policy": legal_normalized_metric["policy"],
        "top_action": top,
        "top_is_outcome_optimal": top in optimal,
        "optimal_mass": legal_normalized_metric["optimal_mass"],
        "degrading_action_mass": legal_normalized_metric["degrading_action_mass"],
        "policy_entropy": legal_normalized_metric["legal_normalized_entropy"],
        "legacy_metric": legacy_metric,
        "legal_normalized_metric": legal_normalized_metric,
        "value_prediction": float(value),
        "exact_wdl_value_error": abs(
            float(value) - float(entry_row["exact_outcome_value"])
        ),
    }


def evaluate_checkpoint(checkpoint: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    rows = []
    for item in manifest["entries"]:
        policy, value = evaluator.evaluate(KalahGame.from_state(item["state"]))
        rows.append(
            {
                "id": item["id"],
                "membership": item["membership"],
                **policy_metrics(policy.tolist(), entry_row=item, value=value),
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
        "rows": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checkpoint_for_generation(
    *, parent: Path, versions_dir: Path, run_id: str, generation: int
) -> Path:
    """Return the checkpoint produced by the normal pipeline naming contract."""
    if generation == 0:
        return parent
    artifact = versions_dir / f"{run_id}-iter{generation}"
    for name in ("checkpoint.npz", "model.npz"):
        candidate = artifact / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing G{generation} checkpoint in {artifact}")


def grouped_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {"cluster": [], "matched_control": []}
    for row in rows:
        groups[
            "matched_control" if row["membership"] == "matched_control" else "cluster"
        ].append(row)
    return {name: aggregate(group) for name, group in groups.items() if group}


def lineage_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_generation = [
        {row["id"]: row for row in evaluation["rows"]} for evaluation in evaluations
    ]
    trajectory_counts: Counter[str] = Counter()
    trajectories: list[dict[str, Any]] = []
    for entry_id in sorted(by_generation[0]):
        rows = [generation[entry_id] for generation in by_generation]
        trajectory = classify_trajectory(
            [bool(row["top_is_outcome_optimal"]) for row in rows]
        )
        trajectory_counts[trajectory] += 1
        trajectories.append(
            {
                "id": entry_id,
                "membership": rows[0]["membership"],
                "trajectory": trajectory,
                "optimal_mass": [row["optimal_mass"] for row in rows],
                "top1_outcome_optimal": [row["top_is_outcome_optimal"] for row in rows],
            }
        )
    masses = [grouped_aggregates(evaluation["rows"]) for evaluation in evaluations]
    return {
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
        "trajectories": trajectories,
        "mass_by_generation": {
            f"G{generation}": value for generation, value in enumerate(masses)
        },
        "final_cluster_delta": masses[-1]["cluster"]["optimal_mass"]
        - masses[0]["cluster"]["optimal_mass"],
        "final_control_delta": masses[-1]["matched_control"]["optimal_mass"]
        - masses[0]["matched_control"]["optimal_mass"],
    }


def run_static_baseline(
    *, checkpoint: Path, baseline_artifact: Path, out: Path, games: int, seed: int
) -> dict[str, Any]:
    """Evaluate a generation against the fixed G0 artifact with the arena CLI."""
    command = [
        sys.executable,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(checkpoint.parent),
        "--current",
        str(baseline_artifact),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return {
        "checkpoint": str(checkpoint),
        "report": str(out),
        "result": json.loads(out.read_text(encoding="utf-8")),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Internal Cluster Learning Dynamics",
        "",
        f"Classification: `{result['classification']}`.",
        "",
        "| seed | G0-G3 cluster optimal-mass delta | G0-G3 control optimal-mass delta | trajectories |",
        "| ---: | ---: | ---: | --- |",
    ]
    for lineage in result["lineages"]:
        if "final_cluster_delta" not in lineage:
            lines.append(f"| {lineage['seed']} | planned | planned | planned |")
            continue
        lines.append(
            f"| {lineage['seed']} | {lineage['final_cluster_delta']:.4f} | "
            f"{lineage['final_control_delta']:.4f} | `{json.dumps(lineage['trajectory_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            f"Frozen set SHA-256: `{result['manifest_sha256']}`.",
            "Each seed starts at the same G0 parent; the frozen set is evaluation-only and is not passed to the pipeline.",
            "The static baseline is the fixed G0 artifact evaluated by the existing `arena.py` CLI; it does not participate in training or promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "optimal_mass": statistics.fmean(row["optimal_mass"] for row in rows),
        "top1_accuracy": statistics.fmean(
            float(row["top_is_outcome_optimal"]) for row in rows
        ),
        "degrading_action_mass": statistics.fmean(
            row["degrading_action_mass"] for row in rows
        ),
        "policy_entropy": statistics.fmean(row["policy_entropy"] for row in rows),
        "value_mae": statistics.fmean(row["exact_wdl_value_error"] for row in rows),
    }


def classify_trajectory(correct: list[bool]) -> str:
    if all(correct):
        return "stable_good"
    changes = sum(a != b for a, b in zip(correct, correct[1:]))
    if not correct[0] and correct[-1] and changes == 1:
        return "repaired"
    if not any(correct):
        return "persistent_failure"
    if changes > 1:
        return "oscillating"
    return "forgotten"


def natural_repair_passes(lineages: list[dict[str, Any]]) -> bool:
    improved = [
        row
        for row in lineages
        if row["final_cluster_delta"] > 0
        and row["final_cluster_delta"] > row["final_control_delta"]
    ]
    meaningful = any(
        row["trajectory_counts"].get("repaired", 0) > 0 for row in lineages
    )
    return (
        len(improved) >= 2
        and meaningful
        and all(not row.get("forensic_regression", False) for row in lineages)
    )


def classify_learning_dynamics(lineages: list[dict[str, Any]]) -> str:
    """Apply the pre-registered outcome labels without selecting a lineage."""
    if natural_repair_passes(lineages):
        return "internal_cluster_naturally_repaired"
    trajectory_sets = {
        tuple(sorted(lineage["trajectory_counts"].items())) for lineage in lineages
    }
    if len(trajectory_sets) > 1:
        return "internal_cluster_learning_dynamics_heterogeneous"
    if any(lineage["trajectory_counts"].get("forgotten", 0) for lineage in lineages):
        return "internal_cluster_forgetting"
    return "internal_cluster_persistent_without_exposure"


def assert_lineage_isolation(lineages: list[dict[str, Any]], parent_sha: str) -> None:
    if any(row["g0_sha256"] != parent_sha for row in lineages):
        raise ValueError("every lineage must start from the same G0 parent")


def build_lineage_config(
    parent: Path, *, seed: int, versions_dir: Path
) -> dict[str, Any]:
    config = json.loads(RECIPE.read_text())
    config.update(
        {
            "run_id": f"internal-cluster-learning-dynamics-seed{seed}",
            "seed": seed,
            "iterations": 3,
            "start_iteration": 1,
            "versions_dir": str(versions_dir),
            "parent_artifact_path": str(parent.parent),
        }
    )
    self_play = next(step for step in config["steps"] if step["name"] == "self_play")
    command = self_play["command"]
    if "--checkpoint" not in command:
        command.extend(["--checkpoint", "{parent_checkpoint}"])
    command[command.index("--seed") + 1] = str(seed)
    command[command.index("--seed-sweep") + 1] = str(seed)
    command[command.index("--total-iterations") + 1] = "3"
    return config


def replay_exposure(replay: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    entries = {canonical_state_key(item["state"]): item for item in manifest["entries"]}
    counts: Counter[str] = Counter()
    target_rows: list[dict[str, Any]] = []
    neighbor_signature = canonical_json(manifest["selected_cluster"])
    neighbors = 0
    with replay.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            state = decode_state(row["state"])
            key = canonical_state_key(state)
            if key in entries:
                counts[entries[key]["membership"]] += 1
                if entries[key]["membership"] in {
                    "cluster_anchor",
                    "structural_cluster",
                }:
                    target_rows.append(row)
            elif cluster_signature(state) == neighbor_signature:
                neighbors += 1
    return {
        "exact_anchor_occurrences": counts["cluster_anchor"],
        "one_ply_parent_occurrences": counts["one_ply_parent"],
        "structural_cluster_occurrences": counts["structural_cluster"],
        "matched_control_occurrences": counts["matched_control"],
        "structural_neighbor_occurrences": neighbors,
        "structural_neighbor_effective_occurrences": neighbors,
        "exact_cluster_targets": [row["policy"] for row in target_rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--uniform-checkpoint", type=Path)
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=ROOT / "storage/ai/alphazero_lite/current",
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--out-result", type=Path)
    parser.add_argument("--out-report", type=Path)
    parser.add_argument("--baseline-games", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.build_manifest:
        if args.control_checkpoint is None or args.uniform_checkpoint is None:
            parser.error(
                "--build-manifest requires historical control and uniform checkpoints"
            )
        pr305 = json.loads(PR305.read_text())
        labels = json.loads(LABELS.read_text())["labels"]
        manifest = build_frozen_set(
            pr305=pr305,
            labels=labels,
            control_evaluator=CheckpointEvaluator(
                args.control_checkpoint, input_encoding="kalah_v3"
            ),
            uniform_evaluator=CheckpointEvaluator(
                args.uniform_checkpoint, input_encoding="kalah_v3"
            ),
        )
        write_json(args.manifest, manifest)
        return 0

    if args.parent_checkpoint is None or args.workdir is None:
        parser.error("learning-dynamics run requires --parent-checkpoint and --workdir")
    if args.out_result is None or args.out_report is None:
        parser.error("learning-dynamics run requires --out-result and --out-report")
    if args.baseline_games < 1:
        parser.error("--baseline-games must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if args.execute and not args.parent_checkpoint.is_file():
        parser.error(f"parent checkpoint does not exist: {args.parent_checkpoint}")
    if args.execute and not args.baseline_artifact.is_dir():
        parser.error(f"baseline artifact does not exist: {args.baseline_artifact}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected frozen manifest schema")
    if manifest.get("training_injection") is not False:
        raise ValueError("frozen manifest must remain outside training")

    lineages: list[dict[str, Any]] = []
    for seed in SEEDS:
        versions_dir = args.workdir / "versions" / f"seed{seed}"
        config = build_lineage_config(
            args.parent_checkpoint, seed=seed, versions_dir=versions_dir
        )
        config_path = args.workdir / "configs" / f"seed{seed}.json"
        write_json(config_path, config)
        lineage: dict[str, Any] = {
            "seed": seed,
            "run_id": config["run_id"],
            "config": str(config_path),
            "pipeline_command": [
                sys.executable,
                "ml/alphazero_lite/pipeline.py",
                "--config",
                str(config_path),
            ],
            "status": "planned",
        }
        if args.execute:
            final_checkpoint = (
                versions_dir / f"{config['run_id']}-iter3" / "checkpoint.npz"
            )
            if not final_checkpoint.exists():
                subprocess.run(lineage["pipeline_command"], cwd=ROOT, check=True)
            evaluations = [
                evaluate_checkpoint(
                    checkpoint_for_generation(
                        parent=args.parent_checkpoint,
                        versions_dir=versions_dir,
                        run_id=config["run_id"],
                        generation=generation,
                    ),
                    manifest,
                )
                for generation in range(4)
            ]
            for generation, evaluation in enumerate(evaluations):
                evaluation["generation"] = f"G{generation}"
            lineage.update(lineage_summary(evaluations))
            lineage["g0_sha256"] = evaluations[0]["checkpoint_sha256"]
            lineage["raw_evaluations"] = evaluations
            lineage["replay_exposure"] = {
                f"G{generation}": replay_exposure(
                    versions_dir
                    / f"{config['run_id']}-iter{generation}"
                    / "self_play.jsonl",
                    manifest,
                )
                for generation in range(1, 4)
            }
            lineage["static_baseline"] = {
                f"G{generation}": run_static_baseline(
                    checkpoint=checkpoint_for_generation(
                        parent=args.parent_checkpoint,
                        versions_dir=versions_dir,
                        run_id=config["run_id"],
                        generation=generation,
                    ),
                    baseline_artifact=args.baseline_artifact,
                    out=args.workdir
                    / "static_baseline"
                    / f"seed{seed}-g{generation}.json",
                    games=args.baseline_games,
                    seed=seed,
                )
                for generation in range(1, 4)
            }
            lineage["status"] = "completed"
        lineages.append(lineage)

    if args.execute:
        parent_sha = sha256_bytes(args.parent_checkpoint.read_bytes())
        assert_lineage_isolation(lineages, parent_sha)
    result = {
        "schema": RESULT_SCHEMA,
        "classification": (
            classify_learning_dynamics(lineages)
            if args.execute
            else "learning_dynamics_planned"
        ),
        "execute": bool(args.execute),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["set_sha256"],
        "parent_checkpoint": str(args.parent_checkpoint),
        "baseline_games": args.baseline_games,
        "lineages": lineages,
    }
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "result": str(args.out_result),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
