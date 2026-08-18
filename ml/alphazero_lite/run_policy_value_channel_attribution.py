#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate PR195 policy/value output composition without training or hybrids.

This runner deliberately combines complete evaluator outputs.  A policy is
always produced by one immutable evaluator and a value by another; no model
parameter, activation, or tensor is copied between checkpoint sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import read_jsonl
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import state_hashes
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    _context_c_puct,
    _search_seed,
    _visit_policy,
    decoded_validation_manifest,
    js,
)
from ml.alphazero_lite.run_policy_target_noise_causal_closeout import (
    forced_continuation,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator, build_eval_search_options

STEPS = (3, 5, 12)
CONTEXTS = ("384:256", "768:768", "1200:1200")
LANES = ("policy_detached_trunk", "heads_only", "baseline_joint")
CANONICAL_ARENA_CONTEXTS = ("384:256", "1200:1200")
CANONICAL_ARENA_MODELS = ("C", "DP_DV", "DP_CV", "CP_DV")
CANONICAL_ARENA_SUITE = Path(
    "/tmp/azlite_shared_trunk_learning/arena_vs_current/temp_0_0/seed_42/"
    "artifact/equal_high/starts_0/opening_suite.jsonl"
)
CANONICAL_ARENA_SEED = 195
CANONICAL_ARENA_OPENINGS = 128
CANONICAL_ARENA_BOOTSTRAP_SAMPLES = 128
DEFAULT_PROBE_WORKERS = 24
WORKDIR = Path("/tmp/azlite_policy_detached_trunk_v2")
MANIFEST = Path("/tmp/azlite_shared_trunk_learning/training_manifest.json")
SUMMARY = (
    REPO_ROOT / "docs/data/alphazero-lite-policy-value-channel-attribution-summary.json"
)
REPORT = REPO_ROOT / "docs/alphazero-lite-policy-value-channel-attribution-results.md"

_PUCT_WORKER_EVALUATORS: dict[str, Any] | None = None
_FORCED_WORKER_EVALUATOR: CheckpointEvaluator | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_state_hashes() -> dict[str, dict[str, str]]:
    """Load the immutable persisted PR195 state-dict hashes before evaluation."""
    source = REPO_ROOT / "docs/data/alphazero-lite-policy-detached-trunk-summary.json"
    return json.loads(source.read_text(encoding="utf-8"))["state_hashes"]


def verify_snapshots(workdir: Path) -> dict[str, dict[str, str]]:
    """Reject missing or altered persisted snapshots; never replay training."""
    expected = expected_state_hashes()
    actual: dict[str, dict[str, str]] = {}
    for lane in LANES:
        snapshots = {
            step: (
                torch.load(
                    workdir / lane / "snapshots" / f"step_{step:04d}.pt",
                    map_location="cpu",
                )["model"],
                {},
            )
            for step in (0, *STEPS)
        }
        actual[lane] = state_hashes(snapshots)
        for step, digest in actual[lane].items():
            if digest != expected[lane][step]:
                raise RuntimeError(f"persisted hash mismatch for {lane} step {step}")
    return actual


def artifact(workdir: Path, lane: str, step: int) -> Path:
    path = workdir / lane / "snapshot_artifacts" / f"step_{step:04d}" / "artifact"
    if not (path / "weights.json").is_file():
        raise FileNotFoundError(f"missing persisted diagnostic artifact: {path}")
    return path


def evaluators(workdir: Path, step: int, current: Path) -> dict[str, Any]:
    """Create isolated output-level compositions for one detached snapshot."""
    detached = artifact(workdir, "policy_detached_trunk", step)
    # Never reuse an evaluator between treatments: PUCT owns mutable search state.
    return {
        "C": arena.ArtifactEvaluator(current),
        "DP_DV": arena.ArtifactEvaluator(detached),
        "DP_CV": arena.ComposedArtifactEvaluator(
            arena.ArtifactEvaluator(current),
            arena.ArtifactEvaluator(detached),
            policy_source="candidate",
            value_source="current",
        ),
        "CP_DV": arena.ComposedArtifactEvaluator(
            arena.ArtifactEvaluator(current),
            arena.ArtifactEvaluator(detached),
            policy_source="current",
            value_source="candidate",
        ),
        "H": arena.ArtifactEvaluator(artifact(workdir, "heads_only", step)),
        "J": arena.ArtifactEvaluator(artifact(workdir, "baseline_joint", step)),
    }


def assert_composition_identity(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    workdir: Path,
    step: int,
    current: Path,
) -> None:
    """Enforce exact whole-output sourcing on every PR195 probe state."""
    current_evaluator = arena.ArtifactEvaluator(current)
    detached_evaluator = arena.ArtifactEvaluator(
        artifact(workdir, "policy_detached_trunk", step)
    )
    for row in rows:
        game = KalahGame.from_state(row["state"])
        current_policy, current_value = current_evaluator.evaluate(game)
        detached_policy, detached_value = detached_evaluator.evaluate(game)
        for label, expected_policy, expected_value in (
            ("C", current_policy, current_value),
            ("DP_DV", detached_policy, detached_value),
            ("DP_CV", detached_policy, current_value),
            ("CP_DV", current_policy, detached_value),
        ):
            policy, value = models[label].evaluate(game)
            if not np.array_equal(policy, expected_policy) or value != expected_value:
                raise AssertionError(
                    f"{label} output composition mismatch at {row['state_hash']}"
                )


def outputs(rows: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    """Compute channel metrics from the exact decoded PR195 validation rows."""
    policies: dict[str, list[np.ndarray]] = {name: [] for name in models}
    values: dict[str, list[float]] = {name: [] for name in models}
    for row in rows:
        game = KalahGame.from_state(row["state"])
        for name, evaluator in models.items():
            policy, value = evaluator.evaluate(game)
            policies[name].append(np.asarray(policy, dtype=float))
            values[name].append(float(value))
    teacher = np.asarray([row["policy"] for row in rows], dtype=float)
    target = np.asarray([row["value"] for row in rows], dtype=float)
    current = np.asarray(policies["C"])
    result: dict[str, Any] = {}
    for name in models:
        policy, value = np.asarray(policies[name]), np.asarray(values[name])
        safe = np.clip(policy, 1e-12, None)
        error = value - target
        result[name] = {
            "policy": {
                "legal_kl_from_current": float(
                    np.mean(
                        np.sum(
                            policy * np.log(safe / np.clip(current, 1e-12, None)),
                            axis=1,
                        )
                    )
                ),
                "replay_teacher_cross_entropy": float(
                    np.mean(-np.sum(teacher * np.log(safe), axis=1))
                ),
                "top1_change_from_current": float(
                    np.mean(np.argmax(policy, axis=1) != np.argmax(current, axis=1))
                ),
                "entropy": float(np.mean(-np.sum(policy * np.log(safe), axis=1))),
            },
            "value": {
                "canonical_outcome_mae": float(np.mean(np.abs(error))),
                "huber_loss": float(
                    np.mean(
                        np.where(np.abs(error) < 1, 0.5 * error**2, np.abs(error) - 0.5)
                    )
                ),
                "sign_accuracy": float(np.mean(np.sign(value) == np.sign(target))),
                "pairwise_concordance": float(
                    np.mean(
                        np.sign(value[1:] - value[:-1])
                        == np.sign(target[1:] - target[:-1])
                    )
                ),
            },
        }
    return result


def _init_puct_worker(workdir: str, step: int, current: str) -> None:
    """Initialize complete, process-local evaluator compositions for one step."""
    global _PUCT_WORKER_EVALUATORS
    _PUCT_WORKER_EVALUATORS = evaluators(Path(workdir), step, Path(current))


def _puct_task(task: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    if _PUCT_WORKER_EVALUATORS is None:
        raise RuntimeError("PUCT worker evaluators are not initialized")
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    result = arena.evaluate_artifact_position(
        evaluator=_PUCT_WORKER_EVALUATORS[task["name"]],
        state=task["state"],
        simulations=task["simulations"],
        seed=task["seed"],
        c_puct=task["c_puct"],
        search_options=options,
    )
    stats = list(result["child_stats"])
    visits = sorted((int(item["visits"]) for item in stats), reverse=True)
    return (
        task["state_hash"],
        task["context"],
        task["name"],
        {
            "move": int(result["selected_move"]),
            "visit": _visit_policy(stats),
            "value": float(result.get("search_root_value", result["value"])),
            "margin": visits[0] - visits[1] if len(visits) > 1 else visits[0],
            "q": {
                int(item["move"]): int(rank)
                for rank, item in enumerate(
                    sorted(
                        stats,
                        key=lambda item: (-float(item["q_value"]), int(item["move"])),
                    ),
                    1,
                )
            },
        },
    )


def puct(
    rows: list[dict[str, Any]],
    *,
    workdir: Path,
    step: int,
    current: Path,
    manifest_hash: str,
    workers: int,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    """Run the fixed deterministic PUCT probe with process-local evaluators."""
    if workers < 1:
        raise ValueError("PUCT workers must be at least one")
    names = ("C", "DP_DV", "DP_CV", "CP_DV", "H", "J")
    tasks = []
    for row in rows[:256]:
        seed, _ = _search_seed(row, manifest_hash, 195)
        for context in CONTEXTS:
            for name in names:
                tasks.append(
                    {
                        "state": row["state"],
                        "state_hash": row["state_hash"],
                        "context": context,
                        "name": name,
                        "simulations": int(context.split(":")[0]),
                        "seed": seed,
                        "c_puct": _context_c_puct(context),
                    }
                )
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_puct_worker,
        initargs=(str(workdir), step, str(current)),
    ) as executor:
        for state_hash, context, name, record in executor.map(_puct_task, tasks):
            records[(state_hash, context, name)] = record
    summary: dict[str, Any] = {}
    for context in CONTEXTS:
        summary[context] = {}
        for name in names:
            pairs = [
                (
                    records[(row["state_hash"], context, "C")],
                    records[(row["state_hash"], context, name)],
                )
                for row in rows[:256]
            ]
            summary[context][name] = {
                "versus": "C",
                "selected_move_change_rate": float(
                    np.mean([a["move"] != b["move"] for a, b in pairs])
                ),
                "visit_js": float(
                    np.mean(
                        [
                            js(np.asarray([a["visit"]]), np.asarray([b["visit"]]))[0]
                            for a, b in pairs
                        ]
                    )
                ),
                "child_q_ranking_changes": float(
                    np.mean(
                        [
                            a["q"].get(b["move"], 0) != b["q"].get(b["move"], 0)
                            for a, b in pairs
                        ]
                    )
                ),
                "root_value_delta": float(
                    np.mean([b["value"] - a["value"] for a, b in pairs])
                ),
                "visit_margin_delta": float(
                    np.mean([b["margin"] - a["margin"] for a, b in pairs])
                ),
            }
    return summary, records


def _init_forced_worker(current_checkpoint: str) -> None:
    global _FORCED_WORKER_EVALUATOR
    _FORCED_WORKER_EVALUATOR = CheckpointEvaluator(
        Path(current_checkpoint), input_encoding="kalah_v3"
    )


def _forced_task(task: dict[str, Any]) -> tuple[float, float]:
    if _FORCED_WORKER_EVALUATOR is None:
        raise RuntimeError("forced-continuation worker evaluator is not initialized")
    control = forced_continuation(
        evaluator=_FORCED_WORKER_EVALUATOR, task=task, forced_move=task["control_move"]
    )
    treated = forced_continuation(
        evaluator=_FORCED_WORKER_EVALUATOR,
        task=task,
        forced_move=task["treatment_move"],
    )
    return (
        (treated["store_margin_root"] - control["store_margin_root"]) / 48.0,
        treated["outcome_root"] - control["outcome_root"],
    )


def forced_audit(
    rows: list[dict[str, Any]],
    records: dict[tuple[str, str, str], dict[str, Any]],
    current_checkpoint: Path,
    workers: int,
) -> dict[str, Any]:
    """Force each qualifying channel treatment choice, then search current/current."""
    if workers < 1:
        raise ValueError("forced-continuation workers must be at least one")
    result: dict[str, Any] = {}
    for context in CONTEXTS:
        for treatment in ("DP_CV", "CP_DV"):
            changed = [
                row
                for row in rows[:256]
                if records[(row["state_hash"], context, "C")]["move"]
                != records[(row["state_hash"], context, treatment)]["move"]
            ]
            if len(changed) < 32:
                continue
            metrics: dict[str, Any] = {
                "changed_roots": len(changed),
                "continuations": {},
            }
            for budget in (768, 1200):
                tasks = []
                for row in changed:
                    seed = stable_seed(
                        "azlite_policy_value_channel_forced",
                        row["state_hash"],
                        context,
                        treatment,
                        budget,
                    )
                    tasks.append(
                        {
                            "state": row["state"],
                            "state_hash": row["state_hash"],
                            "continuation_budget": budget,
                            "experiment_seed": seed,
                            "control_move": records[(row["state_hash"], context, "C")][
                                "move"
                            ],
                            "treatment_move": records[
                                (row["state_hash"], context, treatment)
                            ]["move"],
                        }
                    )
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_forced_worker,
                    initargs=(str(current_checkpoint),),
                ) as executor:
                    continuation_results = list(executor.map(_forced_task, tasks))
                margins = [margin for margin, _ in continuation_results]
                outcomes = [outcome for _, outcome in continuation_results]
                rng = np.random.default_rng(195)
                draw = rng.integers(0, len(margins), size=(10_000, len(margins)))
                margin_draws = np.asarray(margins)[draw].mean(axis=1)
                outcome_draws = np.asarray(outcomes)[draw].mean(axis=1)
                metrics["continuations"][str(budget)] = {
                    "orientation": "treatment-minus-current",
                    "normalized_store_margin_delta": float(np.mean(margins)),
                    "binary_outcome_delta": float(np.mean(outcomes)),
                    "store_margin_bootstrap_95": [
                        float(np.quantile(margin_draws, 0.025)),
                        float(np.quantile(margin_draws, 0.975)),
                    ],
                    "outcome_bootstrap_95": [
                        float(np.quantile(outcome_draws, 0.025)),
                        float(np.quantile(outcome_draws, 0.975)),
                    ],
                    "unique_states": len(changed),
                    "better": int(np.sum(np.asarray(outcomes) > 0)),
                    "worse": int(np.sum(np.asarray(outcomes) < 0)),
                    "tie": int(np.sum(np.asarray(outcomes) == 0)),
                }
            result.setdefault(context, {})[treatment] = metrics
    return result


def canonical_arena_suite(suite_path: Path) -> dict[str, Any]:
    """Verify the fixed, deduplicated 128-opening arena suite."""
    if not suite_path.is_file():
        raise FileNotFoundError(f"missing canonical arena suite: {suite_path}")
    entries = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    openings = [tuple(entry["prefix_moves"]) for entry in entries]
    if (
        len(openings) != CANONICAL_ARENA_OPENINGS
        or len(set(openings)) != CANONICAL_ARENA_OPENINGS
    ):
        raise RuntimeError("canonical arena suite must contain 128 unique openings")
    return {
        "path": str(suite_path),
        "sha256": sha256(suite_path),
        "unique_openings": len(openings),
    }


def complete_arena_records(path: Path) -> list[dict[str, Any]] | None:
    """Return one complete forced-seat arena file, or require it be rerun."""
    if not path.is_file():
        return None
    try:
        records = parse_game_jsonl(str(path))
        opening_indexes = {int(record["opening_index"]) for record in records}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    if (
        len(records) != CANONICAL_ARENA_OPENINGS
        or len(opening_indexes) != CANONICAL_ARENA_OPENINGS
    ):
        return None
    return records


def canonical_arena_matrix(
    *, workdir: Path, current: Path, step: int, workers: int, suite_path: Path
) -> dict[str, Any]:
    """Run the preregistered output-channel arena matrix with matched C/C controls."""
    suite = canonical_arena_suite(suite_path)
    current_artifact = str(current)
    detached_artifact = str(artifact(workdir, "policy_detached_trunk", step))
    model_artifacts = {
        "C": (current_artifact, current_artifact),
        "DP_DV": (detached_artifact, detached_artifact),
        "DP_CV": (detached_artifact, current_artifact),
        "CP_DV": (current_artifact, detached_artifact),
    }
    results: dict[str, Any] = {}
    for context in CANONICAL_ARENA_CONTEXTS:
        challenger_sims, current_sims = (int(value) for value in context.split(":"))
        context_results: dict[str, Any] = {}
        for label in CANONICAL_ARENA_MODELS:
            candidate_records: list[dict[str, Any]] = []
            control_records: list[dict[str, Any]] = []
            policy_artifact, value_artifact = model_artifacts[label]
            evidence_dir = (
                workdir
                / "canonical_arena"
                / f"step_{step:04d}"
                / context.replace(":", "_")
                / label
            )
            for role, challenger_policy, challenger_value, records in (
                ("candidate", policy_artifact, value_artifact, candidate_records),
                (
                    "current_control",
                    current_artifact,
                    current_artifact,
                    control_records,
                ),
            ):
                for seat in (0, 1):
                    seat_dir = evidence_dir / role / f"starts_{seat}"
                    seat_dir.mkdir(parents=True, exist_ok=True)
                    records_path = seat_dir / "games.jsonl"
                    arena_records = complete_arena_records(records_path)
                    if arena_records is None:
                        run_arena(
                            challenger=current_artifact,
                            current=current_artifact,
                            challenger_sims=challenger_sims,
                            current_sims=current_sims,
                            games=CANONICAL_ARENA_OPENINGS,
                            seed=CANONICAL_ARENA_SEED,
                            workers=workers,
                            out_json=str(seat_dir / "arena.json"),
                            out_jsonl=str(records_path),
                            opening_prefixes_jsonl=str(suite_path),
                            challenger_starts=seat,
                            games_per_opening=1,
                            root_policy_mode="deterministic",
                            root_temperature=0.0,
                            normalize_values=False,
                            c_puct=_context_c_puct(context),
                            tactical_root_bias=0.0,
                            challenger_policy_artifact=challenger_policy,
                            challenger_value_artifact=challenger_value,
                            current_policy_artifact=current_artifact,
                            current_value_artifact=current_artifact,
                            seed_contract=SEED_CONTRACT_VERSION,
                            suite_sha256=suite["sha256"],
                            seed_ledger_output=str(seat_dir / "seed_ledger.jsonl"),
                        )
                        arena_records = complete_arena_records(records_path)
                        if arena_records is None:
                            raise RuntimeError(
                                f"incomplete canonical arena evidence: {records_path}"
                            )
                    records.extend(arena_records)
            effect = paired_opening_candidate_effect(
                candidate_records,
                control_records,
                bootstrap_samples=CANONICAL_ARENA_BOOTSTRAP_SAMPLES,
                bootstrap_seed=CANONICAL_ARENA_SEED,
            )
            context_results[label] = {
                "orientation": f"{label}-minus-C",
                "policy_artifact": policy_artifact,
                "value_artifact": value_artifact,
                "paired_candidate_effect": effect["paired_candidate_effect"],
                "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                "orientation_decomposition": effect["orientation_decomposition"],
                "candidate_records_path": str(evidence_dir / "candidate"),
                "current_control_records_path": str(evidence_dir / "current_control"),
            }
        results[context] = context_results
    return {
        "enabled": True,
        "suite": suite,
        "models": list(CANONICAL_ARENA_MODELS),
        "contexts": list(CANONICAL_ARENA_CONTEXTS),
        "search_configuration": {
            "base_seed": CANONICAL_ARENA_SEED,
            "seed_contract": SEED_CONTRACT_VERSION,
            "workers": workers,
            "games_per_opening": 1,
            "forced_challenger_seats": [0, 1],
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "opening_bootstrap_samples": CANONICAL_ARENA_BOOTSTRAP_SAMPLES,
        },
        "metrics": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--arena",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the canonical paired opening-suite arena matrix.",
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--puct-workers", type=int, default=DEFAULT_PROBE_WORKERS)
    args = parser.parse_args()
    hashes = verify_snapshots(args.workdir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = read_jsonl(Path(manifest["replay_path"]))
    indexes = np.load(
        Path(manifest["artifact_paths"]["validation_source_indexes"]),
        allow_pickle=False,
    )
    probe, probe_manifest = decoded_validation_manifest(rows, indexes)
    for row in probe:
        source = rows[row["source_index"]]
        row["policy"] = source["policy"]
        row["value"] = source["value"]
    result: dict[str, Any] = {
        "schema": "azlite_policy_value_channel_attribution_v1",
        "guardrails": {
            "training": False,
            "optimizer_steps": 0,
            "checkpoint_hybrids": False,
        },
        "persisted_state_hashes": hashes,
        "probe_manifest": probe_manifest,
        "steps": {},
    }
    for step in STEPS:
        models = evaluators(args.workdir, step, args.current)
        assert_composition_identity(probe, models, args.workdir, step, args.current)
        if args.puct:
            puct_summary, puct_records = puct(
                probe,
                workdir=args.workdir,
                step=step,
                current=args.current,
                manifest_hash=str(probe_manifest),
                workers=args.puct_workers,
            )
            audit = forced_audit(
                probe,
                puct_records,
                Path("/tmp/azlite_shared_trunk_learning/current.npz"),
                args.puct_workers,
            )
        else:
            puct_summary, audit = {}, {}
        arena_result = (
            canonical_arena_matrix(
                workdir=args.workdir,
                current=args.current,
                step=step,
                workers=args.arena_workers,
                suite_path=CANONICAL_ARENA_SUITE,
            )
            if args.arena
            else {"enabled": False}
        )
        result["steps"][str(step)] = {
            "output_attribution": outputs(probe, models),
            "puct": puct_summary,
            "forced_move_causal_audit": audit,
            "canonical_arena": arena_result,
        }
    SUMMARY.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arena_rows = []
    if args.arena:
        for step, step_result in result["steps"].items():
            for context, metrics in step_result["canonical_arena"]["metrics"].items():
                for label, effect in metrics.items():
                    ci = effect["opening_bootstrap_ci"]
                    arena_rows.append(
                        f"| {step} | {context} | {label} | {effect['paired_candidate_effect']:+.4f} | "
                        f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
                    )
    arena_report = (
        "\n\n## Canonical Arena\n\n"
        "| Step | Budget | Treatment | Paired effect vs C/C | 95% opening bootstrap CI |\n"
        "| ---: | --- | --- | ---: | --- |\n"
        + "\n".join(arena_rows)
        + "\n\nEach treatment uses both forced challenger seats over 128 unique openings; "
        "the matched control is C/C with the same artifact channels, seed contract, and budget.\n"
        if args.arena
        else "\n\n## Canonical Arena\n\nNot run (`--arena` was not supplied).\n"
    )
    REPORT.write_text(
        "# AlphaZero-Lite Policy-Value Channel Attribution\n\n"
        "Generated from immutable evaluator-output composition only. No model was trained or modified."
        + arena_report,
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
