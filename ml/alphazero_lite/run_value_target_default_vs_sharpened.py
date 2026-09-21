#!/usr/bin/env python3
"""Non-promoting paired ablation of fresh default versus sharpened value labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (  # noqa: E402
    evaluate_exact,
    run_arena_pair,
    run_regressions,
)
from ml.alphazero_lite.run_value_head_oracle_audit import (  # noqa: E402
    select_top,
    spearman,
    value_calibration,
)
from ml.alphazero_lite.self_play import outcome_for_player  # noqa: E402

SCHEMA = "azlite_value_target_default_vs_sharpened_v1"
SOURCES = (401, 407, 413, 419, 443, 449)
REPLAY_WEIGHTS = (1, 4, 1, 8, 4)
OPTIMIZER_UPDATES = 1084
BOOTSTRAP_SEED = 353


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def phase(move_index: int) -> str:
    if move_index <= 8:
        return "early"
    if move_index <= 24:
        return "mid"
    return "late"


def target_summary(
    rows: list[tuple[int | None, int, int, float, float]],
) -> dict[str, Any]:
    """Summarise old and default targets without deriving one from the other."""
    old = np.array([item[3] for item in rows], dtype=float)
    default = np.array([item[4] for item in rows], dtype=float)

    def summary(values: np.ndarray) -> dict[str, float | None]:
        if not len(values):
            return {
                "mean": None,
                "mean_absolute": None,
                "median_absolute": None,
                "fraction_absolute_lt_0_25": None,
                "fraction_absolute_lt_0_5": None,
                "fraction_absolute_gt_0_9": None,
            }
        absolute = np.abs(values)
        return {
            "mean": float(values.mean()),
            "mean_absolute": float(absolute.mean()),
            "median_absolute": float(np.median(absolute)),
            "fraction_absolute_lt_0_25": float(np.mean(absolute < 0.25)),
            "fraction_absolute_lt_0_5": float(np.mean(absolute < 0.5)),
            "fraction_absolute_gt_0_9": float(np.mean(absolute > 0.9)),
        }

    outcomes = {"W": 0, "D": 0, "L": 0}
    by_outcome: dict[str, dict[str, list[float]]] = {
        label: {"sharpened": [], "default": []} for label in outcomes
    }
    by_phase: dict[str, dict[str, Any]] = {}
    for winner, player, move_index, _old, _default in rows:
        outcome = outcome_for_player(winner, player)
        outcome_label = {1.0: "W", 0.0: "D", -1.0: "L"}[outcome]
        outcomes[outcome_label] += 1
        by_outcome[outcome_label]["sharpened"].append(float(_old))
        by_outcome[outcome_label]["default"].append(float(_default))
        label = phase(move_index)
        by_phase.setdefault(label, {"sharpened": [], "default": [], "rows": 0})
        by_phase[label]["sharpened"].append(float(_old))
        by_phase[label]["default"].append(float(_default))
        by_phase[label]["rows"] += 1
    return {
        "rows": len(rows),
        "outcome_proportions": {
            key: value / len(rows) for key, value in outcomes.items()
        },
        "target_distribution_by_outcome": {
            label: {
                "rows": outcomes[label],
                "sharpened": summary(np.array(values["sharpened"], dtype=float)),
                "default": summary(np.array(values["default"], dtype=float)),
            }
            for label, values in by_outcome.items()
        },
        "sharpened": summary(old),
        "default": summary(default),
        "by_move_index_bucket": {
            label: {
                "rows": values["rows"],
                "sharpened": summary(np.array(values["sharpened"], dtype=float)),
                "default": summary(np.array(values["default"], dtype=float)),
            }
            for label, values in by_phase.items()
        },
    }


def relabel_row(row: dict[str, Any]) -> dict[str, Any]:
    """Replace only the fresh-row outcome label using the canonical helper."""
    result = dict(row)
    result["value"] = outcome_for_player(row["winner"], row["player"])
    result["value_target_mode"] = "default"
    return result


def verify_relabel_pair(original: dict[str, Any], derived: dict[str, Any]) -> None:
    """Enforce that no training signal except fresh value labels was changed."""
    if set(original) != set(derived):
        raise ValueError("value_target_relabel_integrity_failed")
    if any(
        original[key] != derived[key]
        for key in original
        if key not in {"value", "value_target_mode"}
    ):
        raise ValueError("value_target_relabel_integrity_failed")
    if derived["value"] != outcome_for_player(original["winner"], original["player"]):
        raise ValueError("value_target_relabel_integrity_failed")
    if derived["value_target_mode"] != "default":
        raise ValueError("value_target_relabel_integrity_failed")


def relabel_dataset(source: Path, destination: Path) -> dict[str, Any]:
    """Create and independently audit a deterministic derivative JSONL file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int | None, int, int, float, float]] = []
    with (
        source.open(encoding="utf-8") as input_file,
        destination.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            if not line.strip():
                continue
            original = json.loads(line)
            derived = relabel_row(original)
            verify_relabel_pair(original, derived)
            output_file.write(
                json.dumps(derived, sort_keys=True, ensure_ascii=True) + "\n"
            )
            rows.append(
                (
                    original["winner"],
                    original["player"],
                    int(original["move_index"]),
                    float(original["value"]),
                    float(derived["value"]),
                )
            )
    with (
        source.open(encoding="utf-8") as input_file,
        destination.open(encoding="utf-8") as output_file,
    ):
        source_lines = (line for line in input_file if line.strip())
        derived_lines = (line for line in output_file if line.strip())
        for original_line, derived_line in zip(
            source_lines, derived_lines, strict=True
        ):
            verify_relabel_pair(json.loads(original_line), json.loads(derived_line))
    changed = sum(
        old != default for _winner, _player, _move_index, old, default in rows
    )
    return {
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "row_count": len(rows),
        "rows_value_changed": changed,
        "rows_value_unchanged": len(rows) - changed,
        "only_changed_fields": ["value", "value_target_mode"],
        "integrity": "passed",
        "target_diagnostics": target_summary(rows),
    }


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate all frozen source, replay, parent, and baseline provenance."""
    if (
        plan.get("schema") != SCHEMA
        or tuple(item["seed"] for item in plan["selfplay_datasets"]) != SOURCES
    ):
        raise ValueError("value_target_plan_invalid")
    if tuple(plan["replay_weights"]) != REPLAY_WEIGHTS:
        raise ValueError("value_target_replay_weights_invalid")
    training = plan["training"]
    if (
        training["training_seed"] != 443
        or training["max_optimizer_updates"] != OPTIMIZER_UPDATES
    ):
        raise ValueError("value_target_training_contract_invalid")
    if (
        training["value_target_mode"] != "sharpened"
        or training["policy_target_mode"] != "sharpened"
    ):
        raise ValueError("value_target_baseline_contract_invalid")
    if any(
        plan.get(key) is not False
        for key in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        )
    ):
        raise ValueError("value_target_nonpromotion_contract_invalid")
    parent = ROOT / plan["parent_artifact"] / "weights.json"
    init = ROOT / plan["parent_init_checkpoint"]
    if (
        not parent.is_file()
        or sha256_file(parent) != plan["parent_weights_sha256"]
        or not init.is_file()
    ):
        raise ValueError("value_target_parent_provenance_invalid")
    for replay in plan["replay_sources"]:
        path = ROOT / replay["path"]
        if not path.is_file() or sha256_file(path) != replay["sha256"]:
            raise ValueError("value_target_fixed_replay_provenance_invalid")
    valid = []
    for source in plan["selfplay_datasets"]:
        path = ROOT / source["path"]
        if path.is_file() and sha256_file(path) == source["sha256"]:
            valid.append(source)
    if len(valid) < 5:
        raise ValueError("value_target_source_provenance_unavailable")
    for baseline in plan["sharpened_baselines"]:
        path = Path(baseline["artifact"]) / "checkpoint.npz"
        if not path.is_file() or sha256_file(path) != baseline["checkpoint_sha256"]:
            raise ValueError("value_target_sharpened_baseline_provenance_invalid")
    suite = ROOT / plan["evaluation"]["suite"]
    if not suite.is_file() or sha256_file(suite) != plan["evaluation"]["suite_sha256"]:
        raise ValueError("value_target_diagnostic_suite_unavailable")
    return valid


def training_command(fresh: Path, output: Path, plan: dict[str, Any]) -> list[str]:
    """Render the only permitted treatment optimiser invocation."""
    training = plan["training"]
    files = [fresh, *(ROOT / item["path"] for item in plan["replay_sources"])]
    return [
        str(ROOT / ".venv/bin/python"),
        "ml/alphazero_lite/train.py",
        "--data",
        str(fresh),
        "--data-files",
        ",".join(map(str, files)),
        "--replay-weights",
        ",".join(map(str, REPLAY_WEIGHTS)),
        "--init-checkpoint",
        str(ROOT / plan["parent_init_checkpoint"]),
        "--out",
        str(output),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--device",
        "auto",
        "--lr-scheduler",
        training["lr_scheduler"],
        "--hidden-sizes",
        training["hidden_sizes"],
        "--model-type",
        training["model_type"],
        "--input-encoding",
        training["input_encoding"],
        "--value-loss",
        training["value_loss"],
        "--huber-delta",
        str(training["huber_delta"]),
        "--value-loss-weight",
        str(training["value_loss_weight"]),
        "--val-split",
        str(training["val_split"]),
        "--grad-clip",
        str(training["grad_clip"]),
        "--save-top-k",
        "3",
        "--policy-target-mode",
        training["policy_target_mode"],
        "--value-target-mode",
        "default",
        "--replay-value-target-modes",
        "default,sharpened,sharpened,sharpened,sharpened",
        "--seed",
        str(training["training_seed"]),
        "--max-optimizer-updates",
        str(OPTIMIZER_UPDATES),
        "--final-checkpoint",
        "final",
    ]


def one_ply_and_calibration(
    artifact: Path, corpus: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact)
    calibration_rows, ranking_rows = [], []
    for row in corpus:
        game = KalahGame.from_state(row["canonical_state"])
        priors, root_value = evaluator.evaluate(game)
        legal = [int(move) for move in row["legal_moves"]]
        exact_scores = {
            int(move): float(score)
            for move, score in row["exact"]["exact_score_by_move"].items()
        }
        child_values = {}
        for move in legal:
            child = game.clone()
            child.move(child.pit_index(move))
            _policy, value = evaluator.evaluate(child)
            child_values[move] = (
                float(value)
                if child.current_player == game.current_player
                else -float(value)
            )
        chosen = select_top(
            [child_values.get(move, -float("inf")) for move in range(6)], legal
        )
        pairs = [
            (left, right)
            for left in legal
            for right in legal
            if left < right and exact_scores[left] != exact_scores[right]
        ]
        calibration_rows.append(
            {
                "network_value": float(root_value),
                "exact_outcome": (int(row["exact"]["best_exact_score"]) > 0)
                - (int(row["exact"]["best_exact_score"]) < 0),
                "bucket": row["bucket"],
            }
        )
        ranking_rows.append(
            {
                "top_optimal": chosen
                in [int(move) for move in row["exact"]["exact_optimal_moves"]],
                "regret": float(row["exact"]["exact_regret_by_move"][str(chosen)]),
                "pairwise": None
                if not pairs
                else statistics.fmean(
                    (child_values[a] - child_values[b])
                    * (exact_scores[a] - exact_scores[b])
                    > 0
                    for a, b in pairs
                ),
                "spearman": spearman(child_values, exact_scores),
            }
        )
    return {
        "value_calibration": value_calibration(calibration_rows),
        "value_calibration_by_active_stone_bucket": {
            bucket: value_calibration(
                [row for row in calibration_rows if row["bucket"] == bucket]
            )
            for bucket in sorted({row["bucket"] for row in calibration_rows})
        },
        "one_ply_value_ranking": {
            "top_action_exact_optimal_rate": statistics.fmean(
                row["top_optimal"] for row in ranking_rows
            ),
            "action_ranking_regret": statistics.fmean(
                row["regret"] for row in ranking_rows
            ),
            "pairwise_ranking_accuracy": statistics.fmean(
                row["pairwise"] for row in ranking_rows if row["pairwise"] is not None
            ),
            "spearman_correlation": statistics.fmean(
                row["spearman"] for row in ranking_rows if row["spearman"] is not None
            ),
        },
    }


def evaluate_model(
    artifact: Path, plan: dict[str, Any], regression_path: Path
) -> dict[str, Any]:
    corpus = json.loads((ROOT / plan["exact_corpus"]).read_text(encoding="utf-8"))[
        "states"
    ]
    quality = evaluate_exact(
        artifact,
        {"exact_corpus": {"path": plan["exact_corpus"]}},
        int(plan["evaluation"]["seed"]),
    )
    return {
        **one_ply_and_calibration(artifact, corpus),
        "raw_policy": quality["raw"],
        "mcts_384": quality["mcts_384"],
        "diagnostic_arena": run_arena_pair(
            challenger=artifact,
            current=ROOT / plan["parent_artifact"],
            plan={"evaluation": plan["evaluation"]},
            out_dir=regression_path.parent / "arenas",
            label=artifact.name,
        ),
        "regression": run_regressions(artifact, regression_path),
    }


def paired(values: list[float]) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(10_000)
    )
    return {
        "mean": statistics.fmean(values),
        "bootstrap_95": {
            "seed": BOOTSTRAP_SEED,
            "samples": 10_000,
            "lower": samples[249],
            "upper": samples[9749],
        },
    }


def metric(cell: dict[str, Any], name: str) -> float:
    paths = {
        "root_mae": ("value_calibration", "mae"),
        "sign_accuracy": ("value_calibration", "sign_accuracy"),
        "one_ply_ranking_regret": ("one_ply_value_ranking", "action_ranking_regret"),
        "raw_optimal_mass": ("raw_policy", "optimal_mass"),
        "raw_expected_regret": ("raw_policy", "expected_regret"),
        "mcts_optimal_mass": ("mcts_384", "optimal_mass"),
        "mcts_expected_regret": ("mcts_384", "expected_regret"),
        "arena_score": ("diagnostic_arena", "score"),
    }
    group, key = paths[name]
    return float(cell[group][key])


def classify(
    effects: dict[str, dict[str, Any]], treatment: list[dict[str, Any]]
) -> str:
    improves_mae = effects["root_mae"]["bootstrap_95"]["upper"] < 0
    search_improves = (
        effects["mcts_optimal_mass"]["bootstrap_95"]["lower"] > 0
        or effects["mcts_expected_regret"]["bootstrap_95"]["upper"] < 0
    )
    search_worsens = (
        effects["mcts_optimal_mass"]["bootstrap_95"]["upper"] < 0
        or effects["mcts_expected_regret"]["bootstrap_95"]["lower"] > 0
    )
    raw_regresses = (
        effects["raw_optimal_mass"]["bootstrap_95"]["upper"] < 0
        and effects["raw_expected_regret"]["bootstrap_95"]["lower"] > 0
    )
    arena_bad = effects["arena_score"]["mean"] < 0
    repeated_failures = sum(not cell["regression"]["passed"] for cell in treatment) >= 2
    if (
        improves_mae
        and search_improves
        and not raw_regresses
        and not arena_bad
        and not repeated_failures
    ):
        return "default_value_targets_improve_calibration_and_search"
    if improves_mae and not search_worsens:
        return "default_value_targets_improve_calibration_only"
    if not improves_mae and search_improves:
        return "default_value_targets_improve_search_without_calibration"
    if (not improves_mae and search_worsens) or arena_bad:
        return "sharpened_value_targets_preferred"
    return "value_target_mode_no_clear_benefit"


def aggregate(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Render the preregistered aggregate table from source-level replicates."""
    labels = {
        "root_mae": "root MAE",
        "sign_accuracy": "sign accuracy",
        "one_ply_ranking_regret": "one-ply ranking regret",
        "raw_optimal_mass": "raw optimal mass",
        "raw_expected_regret": "raw expected regret",
        "mcts_optimal_mass": "MCTS optimal mass",
        "mcts_expected_regret": "MCTS expected regret",
        "arena_score": "arena score",
    }
    return {
        name: {
            "metric": labels[name],
            "sharpened": statistics.fmean(metric(cell, name) for cell in baseline),
            "default": statistics.fmean(metric(cell, name) for cell in treatment),
            "paired_delta": effects[name],
        }
        for name in labels
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-value-target-default-vs-sharpened/plan.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/value-target-default-vs-sharpened"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-value-target-default-vs-sharpened/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize:
        result = json.loads(args.out.read_text(encoding="utf-8"))
        result["aggregate"] = aggregate(
            result["sharpened"], result["default"], result["paired_effects"]
        )
        write_json(args.out, result)
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        sources = validate_plan(plan)
        if len(sources) != len(SOURCES):
            raise ValueError("value_target_source_provenance_unavailable")
        manifest = {
            str(source["seed"]): relabel_dataset(
                ROOT / source["path"],
                args.workdir / "relabelled" / f"s{source['seed']}.jsonl",
            )
            for source in sources
        }
    except (FileNotFoundError, ValueError) as error:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": str(error),
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    write_json(
        ROOT
        / "docs/data/alphazero-lite-value-target-default-vs-sharpened/relabel_manifest.json",
        manifest,
    )
    if not args.execute:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": "execution_pending",
                "relabel_manifest": manifest,
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    baselines = {item["source"]: item for item in plan["sharpened_baselines"]}
    baseline_results, treatment_results = [], []
    for source in sources:
        seed = source["seed"]
        baseline_artifact = Path(baselines[seed]["artifact"])
        baseline_results.append(
            {
                "source": seed,
                "artifact": str(baseline_artifact),
                **evaluate_model(
                    baseline_artifact,
                    plan,
                    args.workdir / "regressions" / f"s{seed}-sharpened.json",
                ),
            }
        )
        artifact = args.workdir / "cells" / f"s{seed}-default"
        checkpoint, metrics = (
            artifact / "checkpoint.npz",
            artifact / "training_metrics.json",
        )
        artifact.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            command = training_command(
                args.workdir / "relabelled" / f"s{seed}.jsonl", checkpoint, plan
            ) + ["--training-metrics-out", str(metrics)]
            with (artifact / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(artifact),
                    "--version",
                    f"s{seed}-default",
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
        training_metrics = json.loads(metrics.read_text(encoding="utf-8"))
        if (
            training_metrics["optimizer_updates"] != OPTIMIZER_UPDATES
            or training_metrics["final_checkpoint"] != "final"
        ):
            raise ValueError("value_target_exact_training_contract_failed")
        treatment_results.append(
            {
                "source": seed,
                "artifact": str(artifact),
                "checkpoint_sha256": sha256_file(checkpoint),
                "weights_sha256": sha256_file(artifact / "weights.json"),
                "training": training_metrics,
                **evaluate_model(
                    artifact,
                    plan,
                    args.workdir / "regressions" / f"s{seed}-default.json",
                ),
            }
        )
    names = (
        "root_mae",
        "sign_accuracy",
        "one_ply_ranking_regret",
        "raw_optimal_mass",
        "raw_expected_regret",
        "mcts_optimal_mass",
        "mcts_expected_regret",
        "arena_score",
    )
    effects = {
        name: paired(
            [
                metric(right, name) - metric(left, name)
                for left, right in zip(baseline_results, treatment_results, strict=True)
            ]
        )
        for name in names
    }
    table = [
        {
            "source": left["source"],
            "sharpened_value_mae": metric(left, "root_mae"),
            "default_value_mae": metric(right, "root_mae"),
            "value_mae_delta": metric(right, "root_mae") - metric(left, "root_mae"),
            "sharpened_mcts_mass": metric(left, "mcts_optimal_mass"),
            "default_mcts_mass": metric(right, "mcts_optimal_mass"),
            "mcts_mass_delta": metric(right, "mcts_optimal_mass")
            - metric(left, "mcts_optimal_mass"),
            "arena_delta": metric(right, "arena_score") - metric(left, "arena_score"),
            "regressions": right["regression"],
        }
        for left, right in zip(baseline_results, treatment_results, strict=True)
    ]
    write_json(
        args.out,
        {
            "schema": SCHEMA,
            "plan": plan,
            "relabel_manifest": manifest,
            "sharpened": baseline_results,
            "default": treatment_results,
            "required_table": table,
            "paired_effects": effects,
            "aggregate": aggregate(baseline_results, treatment_results, effects),
            "classification": classify(effects, treatment_results),
            "canonical_gate_run": False,
            "promotion_performed": False,
            "candidate_selection_performed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
