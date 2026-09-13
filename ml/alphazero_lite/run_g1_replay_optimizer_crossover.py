#!/usr/bin/env python3
"""Cross frozen PR #306 G1 replay with explicit training RNG seeds.

This runner intentionally invokes ``train.py`` directly.  It never invokes the
pipeline or self-play, and the PR #305 frozen set is evaluation-only.
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
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (  # noqa: E402
    MANIFEST_SCHEMA,
    approximate_phase,
    canonical_json,
    evaluate_checkpoint,
    grouped_aggregates,
    run_static_baseline,
)
from ml.alphazero_lite.run_internal_state_cluster_provenance_audit import (  # noqa: E402
    cluster_signature,
)

SCHEMA = "azlite_g1_replay_optimizer_crossover_v1"
FROZEN_SET_SHA = "5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a"
G0_SHA = "4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4"
ANCHOR_ID = "cluster_anchor-244dadc26015ebbc"
EXPECTED_REPLAYS = {
    "R61": "6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87",
    "R62": "5cf2f967b16e9c517174a68a0586a3bd9565020ce079809c3fdfc8fc0c61a56e",
    "R63": "f177a20fe7c5ac9a9b50e9c2532571c76f8113c112388aab9a1bd1ba46e0a153",
}
EXPECTED_ORIGINALS = {
    "R61": "c659c26ccd040cb2c8a5b0363fe31573a5d857fd84ac38955ef3d76c8bbeacab",
    "R62": "42c3cbd9e37ae20d9d5b1f201ec0aaaadaa7d5c3d6c3ac9919ad432bcfa7240f",
    "R63": "c58a0ee186d479d32f9182a2dd0b5651db64b9d4f6899481891ff4e4df1827da",
}
TRAINING_SEEDS = {"T61": 61, "T62": 62, "T63": 63}
HISTORICAL_TRAIN_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def artifact_paths() -> dict[str, Any]:
    base = ROOT / ".tmp/internal-cluster-learning-dynamics"
    replays = {
        f"R{seed}": base
        / "versions"
        / f"seed{seed}"
        / f"internal-cluster-learning-dynamics-seed{seed}-iter1"
        / "self_play.jsonl"
        for seed in (61, 62, 63)
    }
    originals = {key: path.with_name("checkpoint.npz") for key, path in replays.items()}
    return {
        "parent": base / "g0-parent/checkpoint.npz",
        "replays": replays,
        "originals": originals,
        "fixed": [
            ROOT
            / "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl",
            ROOT
            / "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl",
        ],
    }


def verify_artifacts(paths: dict[str, Any]) -> dict[str, Any]:
    parent = paths["parent"]
    if not parent.is_file() or sha256_file(parent) != G0_SHA:
        raise RuntimeError(
            "g1_crossover_baseline_not_reproduced: G0 parent unavailable or SHA mismatch"
        )
    replay_shas = {}
    for label, replay in paths["replays"].items():
        if not replay.is_file():
            raise RuntimeError(f"missing frozen dynamic replay: {label}")
        digest = sha256_file(replay)
        if digest != EXPECTED_REPLAYS[label]:
            raise RuntimeError(f"dynamic replay SHA mismatch: {label}")
        replay_shas[label] = digest
    fixed = []
    for source in paths["fixed"]:
        if not source.is_file():
            raise RuntimeError(f"missing fixed replay source: {source}")
        fixed.append({"path": str(source), "sha256": sha256_file(source)})
    for label, checkpoint in paths["originals"].items():
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != EXPECTED_ORIGINALS[label]
        ):
            raise RuntimeError(f"missing or changed original G1 checkpoint: {label}")
    return {"g0_sha256": G0_SHA, "dynamic_replays": replay_shas, "fixed_replays": fixed}


def original_train_config(
    replay: Path, parent: Path, out: Path, *, seed: int, workdir: Path
) -> list[str]:
    """Render the PR #306 train invocation, adding only seed/audit outputs."""
    fixed_a, fixed_b = artifact_paths()["fixed"]
    data_files = ",".join(map(str, (replay, fixed_a, fixed_b)))
    return [
        sys.executable,
        "ml/alphazero_lite/train.py",
        "--data",
        str(replay),
        "--data-files",
        data_files,
        "--replay-weights",
        "1,1,2",
        "--init-checkpoint",
        str(parent),
        "--out",
        str(out),
        "--epochs",
        "4",
        "--batch-size",
        "512",
        "--device",
        "cpu",
        "--lr-scheduler",
        "none",
        "--hidden-sizes",
        "96,3",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--value-loss",
        "huber",
        "--huber-delta",
        "1.0",
        "--value-loss-weight",
        "0.3",
        "--val-split",
        "0.1",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "3",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--seed",
        str(seed),
        "--save-epochs",
        "1,2,3,4",
        "--top-k-dir",
        str(workdir / "epochs"),
        "--epoch-metrics-out",
        str(workdir / "epoch_metrics.json"),
    ]


def compare_training_configs() -> dict[str, Any]:
    """Record actual PR #306 semantics; train.py defaulted to 42 in every lineage."""
    invariant = {
        "parent_checkpoint": G0_SHA,
        "architecture": "residual_v3 hidden_sizes=96,3",
        "encoding": "kalah_v3",
        "epochs": 4,
        "batch_size": 512,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "value_loss_weight": 0.3,
        "policy_target_mode": "sharpened",
        "fixed_replay_weights": [1, 2],
        "checkpoint_selection": "lowest validation loss; save_top_k=3",
        "shuffle": "torch.randperm per epoch",
    }
    return {
        "replay_generating_seed_differences": {"R61": 61, "R62": 62, "R63": 63},
        "training_stochasticity_in_original": {
            "explicit_train_seed": None,
            "train_py_default_seed": HISTORICAL_TRAIN_SEED,
            "rngs": [
                "random",
                "numpy validation source-row split",
                "torch model construction and randperm",
            ],
        },
        "invariant_training_settings": invariant,
    }


def replay_characterization(
    replay: Path, manifest: dict[str, Any], labels: dict[str, Any]
) -> tuple[dict[str, Any], set[int]]:
    entries = {canonical_state_key(item["state"]): item for item in manifest["entries"]}
    signature = canonical_json(manifest["selected_cluster"])
    phases: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    legal: Counter[int] = Counter()
    entropies: list[float] = []
    neighbor_rows: list[dict[str, Any]] = []
    exact_cluster = 0
    parents = 0
    anchors = 0
    states: set[str] = set()
    with replay.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            state = decode_state(row["state"])
            key = canonical_state_key(state)
            states.add(key)
            phases[approximate_phase(state)] += 1
            outcomes[str(row["value"])] += 1
            legal[len(KalahGame.from_state(state).possible_moves())] += 1
            entropies.append(
                -sum(float(p) * math.log2(float(p)) for p in row["policy"] if p > 0)
            )
            membership = entries.get(key, {}).get("membership")
            anchors += int(membership == "cluster_anchor")
            parents += int(membership == "one_ply_parent")
            exact_cluster += int(membership in {"cluster_anchor", "structural_cluster"})
            if membership is None and cluster_signature(state) == signature:
                neighbor_rows.append(row)
    quality = neighbor_target_quality(neighbor_rows, labels)
    return (
        {
            "row_count": len(entropies),
            "unique_canonical_states": len(states),
            "phase_distribution": dict(phases),
            "outcome_distribution": dict(outcomes),
            "policy_entropy": distribution(entropies),
            "legal_action_count_distribution": dict(legal),
            "structural_neighbor_occurrences": len(neighbor_rows),
            "structural_neighbor_effective_occurrences": len(neighbor_rows),
            "exact_anchor_occurrences": anchors,
            "exact_structural_cluster_occurrences": exact_cluster,
            "one_ply_parent_occurrences": parents,
            "structural_neighbor_target_quality": quality,
        },
        {
            index
            for index, row in enumerate(read_rows(replay))
            if cluster_signature(decode_state(row["state"])) == signature
            and canonical_state_key(decode_state(row["state"])) not in entries
        },
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def distribution(values: list[float]) -> dict[str, float]:
    return (
        {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if values
        else {}
    )


def neighbor_target_quality(
    rows: list[dict[str, Any]], labels: dict[str, Any]
) -> dict[str, Any]:
    all_entropy: list[float] = []
    all_max: list[float] = []
    top_actions: Counter[int] = Counter()
    labeled: list[dict[str, float]] = []
    for row in rows:
        policy = [float(p) for p in row["policy"]]
        top = min(range(6), key=lambda action: (-policy[action], action))
        all_entropy.append(-sum(p * math.log2(p) for p in policy if p > 0))
        all_max.append(max(policy))
        top_actions[top] += 1
        label = labels.get(canonical_state_key(decode_state(row["state"])))
        if label is not None:
            optimal = {int(action) for action in label["exact_outcome_optimal_actions"]}
            # PR305 records degrading actions only for the anchor, so do not invent them.
            labeled.append(
                {
                    "top_action_exact_outcome_optimal": float(top in optimal),
                    "optimal_probability_mass": sum(policy[a] for a in optimal),
                }
            )
    return {
        "exact_label_coverage": {"labeled": len(labeled), "total": len(rows)},
        "labeled": {
            "top_action_exact_outcome_optimal_rate": statistics.fmean(
                x["top_action_exact_outcome_optimal"] for x in labeled
            )
            if labeled
            else None,
            "optimal_probability_mass": distribution(
                [x["optimal_probability_mass"] for x in labeled]
            ),
            "degrading_action_probability": "unavailable: no per-neighbor degrading labels",
        },
        "all_rows": {
            "entropy": distribution(all_entropy),
            "max_probability": distribution(all_max),
            "top_action_distribution": dict(top_actions),
        },
    }


def anchor_row(evaluation: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)


def evaluate_epoch_trajectory(
    workdir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    losses = json.loads((workdir / "epoch_metrics.json").read_text())
    rows = []
    for loss in losses:
        checkpoint = workdir / "epochs" / f"checkpoint_epoch{loss['epoch']}.npz"
        metrics = anchor_row(evaluate_checkpoint(checkpoint, manifest))
        rows.append(
            {
                "epoch": loss["epoch"],
                "optimal_mass": metrics["optimal_mass"],
                "action_0_probability": metrics["policy"][0],
                "action_3_probability": metrics["policy"][3],
                "action_4_probability": metrics["policy"][4],
                "top_action": metrics["top_action"],
                "training_policy_loss": loss["policy_loss"],
                "value_loss": loss["value_loss"],
                "structural_neighbor_samples": int(loss["audit_source_samples"] or 0),
            }
        )
    return rows


def run_train(command: list[str], workdir: Path, neighbor_indexes: set[int]) -> None:
    command.extend(
        ["--audit-source-indexes", ",".join(map(str, sorted(neighbor_indexes)))]
    )
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    (workdir / "train.log").write_text(
        result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(f"training failed: {workdir}")


def export_for_arena(checkpoint: Path, artifact_dir: Path, version: str) -> None:
    """Package a local checkpoint for the read-only arena CLI; never promote it."""
    subprocess.run(
        [
            sys.executable,
            "ml/alphazero_lite/export_artifact.py",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(artifact_dir),
            "--version",
            version,
            "--model-type",
            "residual_v3",
            "--rules-version",
            "kalah_v1",
            "--input-encoding",
            "kalah_v3",
        ],
        cwd=ROOT,
        check=True,
    )


def classify(cells: list[dict[str, Any]]) -> str:
    forgotten = sum(bool(cell["anchor_forgotten"]) for cell in cells)
    if forgotten == 0:
        return "anchor_forgetting_not_reproduced"
    by_replay = [
        statistics.fmean(
            c["anchor_optimal_mass_delta"] for c in cells if c["replay"] == replay
        )
        for replay in EXPECTED_REPLAYS
    ]
    by_seed = [
        statistics.fmean(
            c["anchor_optimal_mass_delta"] for c in cells if c["training_seed"] == seed
        )
        for seed in TRAINING_SEEDS
    ]
    replay_range, seed_range = (
        max(by_replay) - min(by_replay),
        max(by_seed) - min(by_seed),
    )
    if replay_range > 1.5 * seed_range:
        return "anchor_forgetting_replay_driven"
    if seed_range > 1.5 * replay_range:
        return "anchor_forgetting_optimizer_driven"
    return "anchor_forgetting_replay_optimizer_interaction"


def anchor_forgotten(g0_anchor: dict[str, Any], g1_anchor: dict[str, Any]) -> bool:
    """The pre-registered binary response; G0 must have begun correct."""
    return bool(g0_anchor["top_is_outcome_optimal"]) and not bool(
        g1_anchor["top_is_outcome_optimal"]
    )


def assert_crossover_isolation(commands: list[list[str]]) -> None:
    """Ensure the generated direct commands differ only in replay and train seed."""
    ignored_values = {
        "--data",
        "--data-files",
        "--out",
        "--seed",
        "--top-k-dir",
        "--epoch-metrics-out",
        "--audit-source-indexes",
    }
    normalized = []
    for command in commands:
        result = []
        skip = False
        for index, token in enumerate(command):
            if skip:
                skip = False
                continue
            if token in ignored_values:
                skip = True
                continue
            result.append(token)
        normalized.append(result)
    if len({tuple(command) for command in normalized}) != 1:
        raise ValueError(
            "crossover commands differ beyond frozen replay and training seed"
        )


def factorial(cells: list[dict[str, Any]]) -> dict[str, Any]:
    values = [cell["anchor_optimal_mass_delta"] for cell in cells]
    grand = statistics.fmean(values)
    replay_means = {
        key: statistics.fmean(
            c["anchor_optimal_mass_delta"] for c in cells if c["replay"] == key
        )
        for key in EXPECTED_REPLAYS
    }
    seed_means = {
        key: statistics.fmean(
            c["anchor_optimal_mass_delta"] for c in cells if c["training_seed"] == key
        )
        for key in TRAINING_SEEDS
    }
    total = sum((value - grand) ** 2 for value in values)
    replay_ss = 3 * sum((value - grand) ** 2 for value in replay_means.values())
    seed_ss = 3 * sum((value - grand) ** 2 for value in seed_means.values())
    interaction_ss = total - replay_ss - seed_ss
    return {
        "grand_mean": grand,
        "replay_means": replay_means,
        "training_seed_means": seed_means,
        "sum_squares": {
            "total": total,
            "replay": replay_ss,
            "training_seed": seed_ss,
            "interaction_residual": interaction_ss,
        },
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# G1 Replay/Optimizer Crossover",
        "",
        f"Classification: `{result['classification']}`.",
        "",
        "## Anchor Optimal-Mass Delta",
        "",
        "| replay | T61 | T62 | T63 |",
        "| --- | ---: | ---: | ---: |",
    ]
    cells = result["cells"]
    for replay in EXPECTED_REPLAYS:
        values = [
            next(
                c for c in cells if c["replay"] == replay and c["training_seed"] == seed
            )["anchor_optimal_mass_delta"]
            for seed in TRAINING_SEEDS
        ]
        lines.append(
            f"| {replay} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Anchor Forgotten",
            "",
            "| replay | T61 | T62 | T63 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for replay in EXPECTED_REPLAYS:
        values = [
            next(
                c for c in cells if c["replay"] == replay and c["training_seed"] == seed
            )["anchor_forgotten"]
            for seed in TRAINING_SEEDS
        ]
        lines.append(
            f"| {replay} | "
            + " | ".join("yes" if value else "no" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- G0 SHA-256: `{G0_SHA}`",
            f"- Frozen set SHA-256: `{FROZEN_SET_SHA}`",
            "- Historical PR #306 training used train.py default seed `42`; T61/T62/T63 explicitly set 61/62/63 and control Python, NumPy validation splitting, Torch initialization RNG consumption, and per-epoch minibatch permutations. Checkpoint loading replaces model initialization weights; there is no dropout or optimizer randomness.",
            "- The runner invokes `train.py` directly only. It does not invoke self-play, pipeline, exact labeling, frozen-set injection, or promotion.",
            "",
            "## Full Record",
            "",
            "Machine-readable details, config comparison, replay characterization, diagonal baseline checks, epoch trajectories, full frozen-set evaluations, counterfactuals, factorial decomposition, and arena safety are in `docs/data/alphazero-lite-g1-replay-optimizer-crossover-results.json`.",
            "",
        ]
    )
    lines.extend(
        [
            "## Inherited Artifacts",
            "",
            *[
                f"- {name} dynamic replay SHA-256: `{digest}`"
                for name, digest in result["inherited_artifacts"][
                    "dynamic_replays"
                ].items()
            ],
            *[
                f"- fixed replay `{row['path']}` SHA-256: `{row['sha256']}`"
                for row in result["inherited_artifacts"]["fixed_replays"]
            ],
            "",
            "## Baseline Reproduction",
            "",
            *[
                f"- {row['replay']} at historical training seed 42: exact checkpoint SHA match `{row['exact_checkpoint_match']}`."
                for row in result.get("baseline_reproduction", [])
            ],
            "",
            "## Replay Characteristics",
            "",
            "| replay | rows | unique states | neighbors | anchor rows | cluster rows | parent rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {name} | {row['row_count']} | {row['unique_canonical_states']} | {row['structural_neighbor_occurrences']} | {row['exact_anchor_occurrences']} | {row['exact_structural_cluster_occurrences']} | {row['one_ply_parent_occurrences']} |"
                for name, row in result["replay_characteristics"].items()
            ],
            "",
            "## Full Frozen Set",
            "",
            "| cell | cluster mass | cluster top-1 | control mass | forgotten | repaired |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {cell['id']} | {cell['frozen_set']['cluster']['optimal_mass']:.4f} | {cell['frozen_set']['cluster']['top1_accuracy']:.4f} | {cell['frozen_set']['matched_control']['optimal_mass']:.4f} | {cell['forgotten_from_g0']} | {cell['repaired_relative_to_g0']} |"
                for cell in cells
            ],
            "",
            "## Epoch Trajectories",
            "",
            "| cell | epoch | mass | P(0) | P(3) | P(4) | top | policy loss | value loss | neighbor samples |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {cell['id']} | {row['epoch']} | {row['optimal_mass']:.4f} | {row['action_0_probability']:.4f} | {row['action_3_probability']:.4f} | {row['action_4_probability']:.4f} | {row['top_action']} | {row['training_policy_loss']:.4f} | {row['value_loss']:.4f} | {row['structural_neighbor_samples']} |"
                for cell in cells
                for row in cell["epoch_trajectory"]
            ],
            "",
            "## Effects, Interaction, And Safety",
            "",
            f"- Replay effect: `{json.dumps(result['replay_aggregation'], sort_keys=True)}`",
            f"- Training-seed effect: `{json.dumps(result['training_seed_aggregation'], sort_keys=True)}`",
            f"- Factorial decomposition: `{json.dumps(result['factorial_decomposition'], sort_keys=True)}`",
            f"- Structural-neighbor target characterization: `{json.dumps({name: row['structural_neighbor_target_quality'] for name, row in result['replay_characteristics'].items()}, sort_keys=True)}`",
            f"- Lightweight fixed-arena safety: `{json.dumps(result.get('arena_safety', {}), sort_keys=True)}`",
            f"- Exactly one next experiment: {result.get('next_experiment', 'not selected')}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    paths = artifact_paths()
    inherited = verify_artifacts(paths)
    manifest_path = (
        ROOT / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("set_sha256") != FROZEN_SET_SHA
        or manifest.get("training_injection") is not False
    ):
        raise RuntimeError("frozen set identity/injection guard failed")
    labels = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit-labels.json"
        ).read_text()
    )["labels"]
    characteristics, neighbor_indexes = {}, {}
    for replay, path in paths["replays"].items():
        characteristics[replay], neighbor_indexes[replay] = replay_characterization(
            path, manifest, labels
        )
        if characteristics[replay]["exact_anchor_occurrences"] != 0:
            raise RuntimeError("frozen anchor occurs in replay")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "execute": args.execute,
        "inherited_artifacts": inherited,
        "config_diff": compare_training_configs(),
        "frozen_set_sha256": FROZEN_SET_SHA,
        "replay_characteristics": characteristics,
        "training_seed_semantics": {"T61": 61, "T62": 62, "T63": 63},
        "cells": [],
    }
    if not args.execute:
        result["classification"] = "planned"
        write_json(args.out_result, result)
        args.out_report.write_text(render_report(result), encoding="utf-8")
        return 0
    g0_evaluation = evaluate_checkpoint(paths["parent"], manifest)
    g0_anchor = anchor_row(g0_evaluation)
    g0_frozen = grouped_aggregates(g0_evaluation["rows"])
    if not g0_anchor["top_is_outcome_optimal"]:
        raise RuntimeError("G0 anchor is not exact-outcome optimal")
    baseline = []
    for replay, replay_path in paths["replays"].items():
        workdir = args.workdir / "baseline" / replay
        output = workdir / "checkpoint.npz"
        run_train(
            original_train_config(
                replay_path,
                paths["parent"],
                output,
                seed=HISTORICAL_TRAIN_SEED,
                workdir=workdir,
            ),
            workdir,
            neighbor_indexes[replay],
        )
        actual_sha = sha256_file(output)
        baseline.append(
            {
                "replay": replay,
                "historical_training_seed": HISTORICAL_TRAIN_SEED,
                "expected_checkpoint_sha256": EXPECTED_ORIGINALS[replay],
                "actual_checkpoint_sha256": actual_sha,
                "exact_checkpoint_match": actual_sha == EXPECTED_ORIGINALS[replay],
            }
        )
    if not all(row["exact_checkpoint_match"] for row in baseline):
        result.update(
            {
                "baseline_reproduction": baseline,
                "classification": "anchor_forgetting_crossover_inconclusive",
            }
        )
        write_json(args.out_result, result)
        args.out_report.write_text(render_report(result), encoding="utf-8")
        raise RuntimeError("g1_crossover_baseline_not_reproduced")
    result["baseline_reproduction"] = baseline
    for replay, replay_path in paths["replays"].items():
        for training_seed, seed in TRAINING_SEEDS.items():
            workdir = args.workdir / "cells" / f"{replay}-{training_seed}"
            output = workdir / "checkpoint.npz"
            run_train(
                original_train_config(
                    replay_path, paths["parent"], output, seed=seed, workdir=workdir
                ),
                workdir,
                neighbor_indexes[replay],
            )
            evaluation = evaluate_checkpoint(output, manifest)
            anchor = anchor_row(evaluation)
            trajectory = evaluate_epoch_trajectory(workdir, manifest)
            cell = {
                "id": f"{replay}-{training_seed}",
                "replay": replay,
                "training_seed": training_seed,
                "seed": seed,
                "g0_sha256": sha256_file(paths["parent"]),
                "checkpoint_sha256": sha256_file(output),
                "anchor": anchor,
                "anchor_optimal_mass_delta": anchor["optimal_mass"]
                - g0_anchor["optimal_mass"],
                "anchor_forgotten": anchor_forgotten(g0_anchor, anchor),
                "frozen_set": grouped_aggregates(evaluation["rows"]),
                "forgotten_from_g0": sum(
                    not row["top_is_outcome_optimal"]
                    for row in evaluation["rows"]
                    if row["membership"] != "matched_control"
                ),
                "repaired_relative_to_g0": 0,
                "per_state_correctness": {
                    row["id"]: row["top_is_outcome_optimal"]
                    for row in evaluation["rows"]
                },
                "epoch_trajectory": trajectory,
                "minibatch_exposure": {
                    "source_identity": replay,
                    "samples_consumed": sum(
                        x["structural_neighbor_samples"] for x in trajectory
                    ),
                    "first_exposure_epoch": next(
                        (
                            x["epoch"]
                            for x in trajectory
                            if x["structural_neighbor_samples"]
                        ),
                        None,
                    ),
                    "per_epoch": [x["structural_neighbor_samples"] for x in trajectory],
                },
            }
            result["cells"].append(cell)
    assert_crossover_isolation(
        [
            original_train_config(
                paths["replays"][cell["replay"]],
                paths["parent"],
                args.workdir / "verify.npz",
                seed=cell["seed"],
                workdir=args.workdir / "verify",
            )
            for cell in result["cells"]
        ]
    )
    result["g0_anchor"] = g0_anchor
    result["g0_frozen_set"] = g0_frozen
    result["replay_aggregation"] = {
        replay: {
            "mean_anchor_optimal_mass_delta": statistics.fmean(
                c["anchor_optimal_mass_delta"]
                for c in result["cells"]
                if c["replay"] == replay
            ),
            "median_anchor_optimal_mass_delta": statistics.median(
                c["anchor_optimal_mass_delta"]
                for c in result["cells"]
                if c["replay"] == replay
            ),
            "anchor_forgotten_count": sum(
                c["anchor_forgotten"] for c in result["cells"] if c["replay"] == replay
            ),
            "mean_full_cluster_delta": statistics.fmean(
                c["frozen_set"]["cluster"]["optimal_mass"]
                - g0_frozen["cluster"]["optimal_mass"]
                for c in result["cells"]
                if c["replay"] == replay
            ),
        }
        for replay in EXPECTED_REPLAYS
    }
    result["training_seed_aggregation"] = {
        seed: {
            "mean_anchor_optimal_mass_delta": statistics.fmean(
                c["anchor_optimal_mass_delta"]
                for c in result["cells"]
                if c["training_seed"] == seed
            ),
            "anchor_forgotten_count": sum(
                c["anchor_forgotten"]
                for c in result["cells"]
                if c["training_seed"] == seed
            ),
        }
        for seed in TRAINING_SEEDS
    }
    result["factorial_decomposition"] = factorial(result["cells"])
    baseline_artifact = ROOT / "storage/ai/alphazero_lite/current"
    for cell in result["cells"]:
        export_for_arena(
            args.workdir / "cells" / cell["id"] / "checkpoint.npz",
            args.workdir / "cells" / cell["id"],
            cell["id"],
        )
    result["arena_safety"] = {
        cell["id"]: run_static_baseline(
            checkpoint=args.workdir / "cells" / cell["id"] / "checkpoint.npz",
            baseline_artifact=baseline_artifact,
            out=args.workdir / "arena" / f"{cell['id']}.json",
            games=30,
            seed=cell["seed"],
        )["result"]
        for cell in result["cells"]
    }
    result["classification"] = classify(result["cells"])
    result["next_experiment"] = {
        "anchor_forgetting_replay_driven": "perform a replay-content difference audit between the protective and harmful corpus focused on structural-neighbor target families; do not train yet.",
        "anchor_forgetting_optimizer_driven": "run a training-order/minibatch-interference diagnostic under one frozen replay corpus.",
        "anchor_forgetting_replay_optimizer_interaction": "freeze the most diagnostic replay pair and trace minibatch-level anchor policy drift to locate interaction events.",
        "anchor_forgetting_not_reproduced": "expand the frozen state subcluster before changing training.",
    }[result["classification"]]
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
