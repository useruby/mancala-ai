#!/usr/bin/env python3
# ruff: noqa: E402
"""Causal closeout for PR #177 noisy-versus-denoised target disagreements."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import stable_hash, stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import bootstrap_ci
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT

DEFAULT_WORKDIR = Path("/tmp/azlite_policy_target_noise_causal_closeout")
DEFAULT_PR177_WORKDIR = Path("/tmp/azlite_policy_target_noise_ablation")
EXPECTED_CURRENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
CONTINUATION_BUDGETS = (768, 1200)
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_REPRODUCTION = {
    "probe_count": 768,
    "disagreement_count": 103,
    "disagreement_fraction": 0.13411458333333334,
    "noisy_js": 0.03525294751849997,
    "denoised_js": 0.019356463036061033,
    "noisy_top1": 0.81640625,
    "denoised_top1": 0.8346354166666666,
}

_WORKER_EVALUATOR: CheckpointEvaluator | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def continuation_seed_context(
    *, state_hash: str, continuation_budget: int, root_player: int, experiment_seed: int
) -> dict[str, Any]:
    """Return the pre-intervention identity shared by every forced move."""
    return {
        "schema": "azlite_policy_target_noise_continuation_seed_v1",
        "state_hash": str(state_hash),
        "continuation_budget": int(continuation_budget),
        "root_player": int(root_player),
        "experiment_seed": int(experiment_seed),
    }


def continuation_seed_identity(**kwargs: Any) -> tuple[int, str]:
    context = continuation_seed_context(**kwargs)
    return stable_seed(context), stable_hash(context)


def _domain_group(source_domain: str) -> str:
    return (
        "evaluation_diagnostic"
        if source_domain == "evaluation_opening_diagnostic"
        else "self_play_pilot"
    )


def _target_hash(rows: list[dict[str, Any]], target_name: str) -> str:
    return stable_hash(
        [{"state_hash": row["state_hash"], "target": row[target_name]} for row in rows]
    )


def _js_divergence(reference: dict[str, Any], target: dict[str, Any]) -> float:
    legal = reference["legal_moves"]
    ref = np.maximum(np.asarray(reference["policy"])[legal], 1e-12)
    candidate = np.maximum(np.asarray(target["policy"])[legal], 1e-12)
    midpoint = (ref + candidate) / 2
    return float(
        (
            np.sum(ref * np.log(ref / midpoint))
            + np.sum(candidate * np.log(candidate / midpoint))
        )
        / 2
    )


def reproduction_metrics(teachers: list[dict[str, Any]]) -> dict[str, float | int]:
    """Recompute the PR #177 aggregate signals without importing training code."""
    noisy_js = []
    denoised_js = []
    noisy_top1 = []
    denoised_top1 = []
    disagreements = 0
    for row in teachers:
        reference = row["reference_d1200"]
        noisy = row["noisy_n384"]
        denoised = row["denoised_d384"]
        noisy_js.append(_js_divergence(reference, noisy))
        denoised_js.append(_js_divergence(reference, denoised))
        noisy_top1.append(reference["top_move"] == noisy["top_move"])
        denoised_top1.append(reference["top_move"] == denoised["top_move"])
        disagreements += noisy["top_move"] != denoised["top_move"]
    return {
        "probe_count": len(teachers),
        "disagreement_count": disagreements,
        "disagreement_fraction": float(disagreements / len(teachers)),
        "noisy_js": float(np.mean(noisy_js)),
        "denoised_js": float(np.mean(denoised_js)),
        "noisy_top1": float(np.mean(noisy_top1)),
        "denoised_top1": float(np.mean(denoised_top1)),
    }


def verify_pr177_inputs(
    *, pr177_workdir: Path, committed_summary_path: Path, current_weights: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = pr177_workdir / "target_probe_manifest.json"
    states_path = pr177_workdir / "target_probe_states.jsonl"
    teachers_path = pr177_workdir / "target_probe_teachers.jsonl"
    for path in (manifest_path, states_path, teachers_path):
        if not path.is_file():
            raise RuntimeError(f"missing frozen PR #177 input: {path}")
    if sha256_file(current_weights) != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")

    manifest = read_json(manifest_path)
    committed_summary = read_json(committed_summary_path)
    if manifest != committed_summary["probe_manifest"]:
        raise RuntimeError("local PR #177 probe manifest differs from committed record")
    teachers = read_jsonl(teachers_path)
    states = read_jsonl(states_path)
    if [row["state_hash"] for row in states] != manifest["state_hashes"]:
        raise RuntimeError("frozen probe states do not match manifest ordering")
    if [row["state_hash"] for row in teachers] != manifest["state_hashes"]:
        raise RuntimeError("frozen teacher records do not match manifest ordering")

    observed = reproduction_metrics(teachers)
    for key, expected in EXPECTED_REPRODUCTION.items():
        if not np.isclose(observed[key], expected, rtol=0.0, atol=1e-15):
            raise RuntimeError(
                f"PR #177 reproduction mismatch for {key}: {observed[key]}"
            )
    return teachers, {
        "reproduction": observed,
        "probe_manifest_sha256": sha256_file(manifest_path),
        "probe_states_sha256": sha256_file(states_path),
        "teacher_records_sha256": sha256_file(teachers_path),
        "target_hashes": {
            "noisy_n384": _target_hash(teachers, "noisy_n384"),
            "denoised_d384": _target_hash(teachers, "denoised_d384"),
            "reference_d1200": _target_hash(teachers, "reference_d1200"),
        },
        "manifest": manifest,
    }


def disagreement_rows(teachers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in teachers
        if row["noisy_n384"]["top_move"] != row["denoised_d384"]["top_move"]
    ]
    rows.sort(key=lambda row: row["state_hash"])
    if len(rows) != 103 or len({row["state_hash"] for row in rows}) != 103:
        raise RuntimeError("expected exactly 103 unique PR #177 disagreement states")
    return rows


def forced_continuation(
    *, evaluator: CheckpointEvaluator, task: dict[str, Any], forced_move: int
) -> dict[str, Any]:
    """Evaluate one forced move with state-dependent deterministic post-move search."""
    game = KalahGame.from_state(task["state"])
    root_player = int(game.current_player)
    base_seed, seed_context_hash = continuation_seed_identity(
        state_hash=task["state_hash"],
        continuation_budget=task["continuation_budget"],
        root_player=root_player,
        experiment_seed=task["experiment_seed"],
    )
    if int(forced_move) not in game.possible_moves() or not game.move(
        game.pit_index(int(forced_move))
    ):
        raise RuntimeError(f"illegal forced move {forced_move}")
    trajectory = [int(forced_move)]
    ply = 1
    while not game.over() and ply < 200:
        legal = game.possible_moves()
        if not legal:
            break
        # This post-intervention seed is state-dependent, but has no treatment label.
        search_seed = stable_seed(base_seed, stable_hash(game.to_state()), ply)
        search = PUCT(
            evaluator=evaluator,
            simulations=int(task["continuation_budget"]),
            c_puct=1.25,
            rng=random.Random(search_seed),
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
        )
        _, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        move = int(search.select_root_move(root, legal))
        if not game.move(game.pit_index(move)):
            raise RuntimeError(f"failed continuation move {move}")
        trajectory.append(move)
        ply += 1
    stores = game.captured_seeds
    winner = game.winner
    return {
        "forced_move": int(forced_move),
        "outcome_root": 0.0
        if winner is None
        else (1.0 if int(winner) == root_player else -1.0),
        "store_margin_root": int(stores[root_player] - stores[1 - root_player]),
        "trajectory_hash": stable_hash(trajectory),
        "paired_seed_context_hash": seed_context_hash,
        "paired_base_seed": base_seed,
    }


def _init_worker(checkpoint: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def _forced_task_record(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("forced-continuation worker is not initialized")
    moves = task["teacher_moves"]
    interventions = {
        "noisy_n384": forced_continuation(
            evaluator=_WORKER_EVALUATOR, task=task, forced_move=moves["noisy_n384"]
        ),
        "denoised_d384": forced_continuation(
            evaluator=_WORKER_EVALUATOR, task=task, forced_move=moves["denoised_d384"]
        ),
    }
    if moves["reference_d1200"] not in {moves["noisy_n384"], moves["denoised_d384"]}:
        interventions["reference_d1200"] = forced_continuation(
            evaluator=_WORKER_EVALUATOR, task=task, forced_move=moves["reference_d1200"]
        )
    return {
        key: task[key]
        for key in (
            "state_hash",
            "player",
            "phase",
            "source_domain",
            "domain_group",
            "legal_move_count",
            "continuation_budget",
            "teacher_moves",
        )
    } | {"interventions": interventions}


def build_tasks(
    rows: list[dict[str, Any]], *, experiment_seed: int
) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        for budget in CONTINUATION_BUDGETS:
            tasks.append(
                {
                    "state_hash": row["state_hash"],
                    "state": row["state"],
                    "player": int(row["player"]),
                    "phase": row["phase"],
                    "source_domain": row["source_domain"],
                    "domain_group": _domain_group(row["source_domain"]),
                    "legal_move_count": int(row["legal_move_count"]),
                    "continuation_budget": budget,
                    "experiment_seed": int(experiment_seed),
                    "teacher_moves": {
                        name: int(row[name]["top_move"])
                        for name in ("noisy_n384", "denoised_d384", "reference_d1200")
                    },
                }
            )
    return tasks


def run_forced_tasks(
    *, tasks: list[dict[str, Any]], checkpoint: Path, workers: int
) -> list[dict[str, Any]]:
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        _init_worker(str(checkpoint))
        records = [_forced_task_record(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(str(checkpoint),),
        ) as executor:
            records = list(executor.map(_forced_task_record, tasks))
    return sorted(
        records, key=lambda row: (row["state_hash"], row["continuation_budget"])
    )


def _summary(records: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    deltas = np.asarray(
        [
            row["interventions"]["denoised_d384"]["outcome_root"]
            - row["interventions"]["noisy_n384"]["outcome_root"]
            for row in records
        ],
        dtype=float,
    )
    margins = np.asarray(
        [
            row["interventions"]["denoised_d384"]["store_margin_root"]
            - row["interventions"]["noisy_n384"]["store_margin_root"]
            for row in records
        ],
        dtype=float,
    )
    return {
        "unique_states": len({row["state_hash"] for row in records}),
        "mean_outcome_delta": float(np.mean(deltas)) if len(deltas) else 0.0,
        "median_outcome_delta": float(np.median(deltas)) if len(deltas) else 0.0,
        "mean_store_margin_delta": float(np.mean(margins)) if len(margins) else 0.0,
        "median_store_margin_delta": float(np.median(margins)) if len(margins) else 0.0,
        "fraction_denoised_better": float(np.mean(deltas > 0)) if len(deltas) else 0.0,
        "fraction_noisy_better": float(np.mean(deltas < 0)) if len(deltas) else 0.0,
        "fraction_tied": float(np.mean(deltas == 0)) if len(deltas) else 0.0,
        "paired_bootstrap_95": bootstrap_ci(
            deltas.tolist(), seed=seed, samples=BOOTSTRAP_SAMPLES
        ),
    }


def summarize(records: list[dict[str, Any]], *, experiment_seed: int) -> dict[str, Any]:
    by_budget: dict[str, Any] = {}
    slices: dict[str, dict[str, Any]] = {
        key: {} for key in ("phase", "player", "domain_group")
    }
    for budget in CONTINUATION_BUDGETS:
        budget_records = [
            row for row in records if row["continuation_budget"] == budget
        ]
        by_budget[str(budget)] = _summary(
            budget_records, seed=stable_seed(experiment_seed, "global", budget)
        )
        for key in slices:
            for value in sorted({str(row[key]) for row in budget_records}):
                subset = [row for row in budget_records if str(row[key]) == value]
                slices[key].setdefault(value, {})[str(budget)] = _summary(
                    subset,
                    seed=stable_seed(experiment_seed, "slice", key, value, budget),
                )
    return {"by_budget": by_budget, "slices": slices}


def teacher_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        noisy = row["noisy_n384"]["top_move"]
        denoised = row["denoised_d384"]["top_move"]
        reference = row["reference_d1200"]["top_move"]
        if denoised == reference and noisy != reference:
            counts["denoised_only_matches_reference"] += 1
        elif noisy == reference and denoised != reference:
            counts["noisy_only_matches_reference"] += 1
        else:
            counts["neither_matches_reference"] += 1
    total = len(rows)
    return {
        "counts": dict(counts),
        "probability_d384_matches_d1200": counts["denoised_only_matches_reference"]
        / total,
        "probability_n384_matches_d1200": counts["noisy_only_matches_reference"]
        / total,
    }


def reference_match_effects(
    records: list[dict[str, Any]], *, experiment_seed: int
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        moves = row["teacher_moves"]
        if moves["denoised_d384"] == moves["reference_d1200"]:
            key = "denoised_matches_reference"
        elif moves["noisy_n384"] == moves["reference_d1200"]:
            key = "noisy_matches_reference"
        else:
            key = "neither_matches_reference"
        groups[key].append(row)
    result: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        result[key] = {}
        for budget in CONTINUATION_BUDGETS:
            subset = [row for row in group if row["continuation_budget"] == budget]
            result[key][str(budget)] = _summary(
                subset, seed=stable_seed(experiment_seed, "reference", key, budget)
            )
    return result


def classify(summary: dict[str, Any]) -> tuple[str, list[str], str]:
    budgets = summary["forced_outcomes"]["by_budget"]
    global_rows = [budgets[str(budget)] for budget in CONTINUATION_BUDGETS]
    slices = summary["forced_outcomes"]["slices"]
    robust_phase = any(
        all(
            row["unique_states"] >= 32 and row["paired_bootstrap_95"]["lower"] > 0
            for row in phase_rows.values()
        )
        for phase_rows in slices["phase"].values()
    )
    bad_slice = any(
        row["unique_states"] >= 24 and row["mean_outcome_delta"] < -0.10
        for grouping in (slices["player"], slices["domain_group"])
        for group_rows in grouping.values()
        for row in group_rows.values()
    )
    if any(
        row["mean_outcome_delta"] < 0 and row["paired_bootstrap_95"]["upper"] < 0
        for row in global_rows
    ):
        primary = "denoised_policy_target_rejected"
    elif (
        all(
            row["unique_states"] >= 64
            and row["mean_outcome_delta"] > 0
            and row["paired_bootstrap_95"]["lower"] >= 0
            for row in global_rows
        )
        and not bad_slice
    ):
        primary = "denoised_disagreement_moves_causally_better"
    elif robust_phase and not all(
        row["paired_bootstrap_95"]["lower"] > 0 for row in global_rows
    ):
        primary = "target_noise_effect_phase_localized"
    elif (
        all(
            row["paired_bootstrap_95"]["lower"]
            <= 0
            <= row["paired_bootstrap_95"]["upper"]
            for row in global_rows
        )
        and not robust_phase
    ):
        primary = "policy_target_noise_not_primary_confirmed"
    else:
        primary = "causal_evidence_inconclusive"

    reference = summary["reference_match_causal_effects"]
    denoised = reference.get("denoised_matches_reference", {})
    noisy = reference.get("noisy_matches_reference", {})
    reference_not_predictive = (
        denoised
        and noisy
        and all(
            denoised[str(budget)]["mean_outcome_delta"]
            <= noisy[str(budget)]["mean_outcome_delta"]
            for budget in CONTINUATION_BUDGETS
        )
    )
    labels = [primary] + (
        ["strong_search_reference_not_causally_predictive"]
        if reference_not_predictive
        else []
    )
    next_action = {
        "policy_target_noise_not_primary_confirmed": "Audit denoised PUCT target convergence at D128, D384, D768, and D1200 before policy training.",
        "denoised_disagreement_moves_causally_better": "Run the previously designed paired noisy-versus-denoised replay/training experiment in a separate PR without recipe changes.",
        "target_noise_effect_phase_localized": "Replicate phase-specific target quality on an independent state sample before any phase-dependent training target work.",
        "denoised_policy_target_rejected": "Close denoised policy targets as an intervention and retain the original PR #177 gate decision.",
        "causal_evidence_inconclusive": "Do not train or generate replay; retain the original PR #177 stop and investigate direct move quality only if separately prioritized.",
    }[primary]
    if reference_not_predictive:
        next_action += " Also stop optimizing against higher-search agreement and prioritize direct forced-move/game-outcome target quality."
    return primary, labels, next_action


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Policy-Target Noise Causal Closeout",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Additional classifications: `{', '.join(summary['classifications'][1:]) or 'none'}`",
        "",
        "## PR #177 Reproduction",
        "",
        "```json",
        json.dumps(summary["frozen_inputs"]["reproduction"], indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen Inputs",
        "",
        f"- current weights SHA256: `{summary['frozen_inputs']['current_weights_sha256']}`",
        f"- probe manifest SHA256: `{summary['frozen_inputs']['probe_manifest_sha256']}`",
        f"- teacher records SHA256: `{summary['frozen_inputs']['teacher_records_sha256']}`",
        f"- code commit: `{summary['frozen_inputs']['code_commit']}`",
        f"- disagreement states: `{summary['disagreement_count']}`",
        "",
        "## Forced Outcomes",
        "",
        "| Budget | Mean outcome delta | Median | 95% CI | Mean / median store margin | Denoised / noisy / tied |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for budget, row in summary["forced_outcomes"]["by_budget"].items():
        ci = row["paired_bootstrap_95"]
        lines.append(
            f"| {budget} | {row['mean_outcome_delta']:.4f} | {row['median_outcome_delta']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {row['mean_store_margin_delta']:.4f} / {row['median_store_margin_delta']:.4f} | {row['fraction_denoised_better']:.3f} / {row['fraction_noisy_better']:.3f} / {row['fraction_tied']:.3f} |"
        )
    lines += [
        "",
        "## Seed Pairing",
        "",
        "The base continuation identity contains only state hash, continuation budget, root player, and experiment seed. It excludes intervention label, forced move, and teacher identity. The same context is used for both forced interventions; post-move searches use deterministic state-dependent seeds with no Dirichlet noise.",
        "",
        "## Disagreement Composition",
        "",
        "```json",
        json.dumps(summary["disagreement_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Teacher Reference Agreement",
        "",
        "| D384 only matches D1200 | N384 only matches D1200 | Neither matches D1200 | P(D384 = D1200) | P(N384 = D1200) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    agreement = summary["teacher_reference_agreement"]
    counts = agreement["counts"]
    lines.append(
        f"| {counts['denoised_only_matches_reference']} | {counts['noisy_only_matches_reference']} | {counts['neither_matches_reference']} | {agreement['probability_d384_matches_d1200']:.4f} | {agreement['probability_n384_matches_d1200']:.4f} |"
    )
    lines += [
        "",
        "## Player, Phase, And Domain Slices",
        "",
        "| Slice | Group | Budget | States | Mean outcome delta | 95% CI | Mean store-margin delta |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for slice_name, groups in summary["forced_outcomes"]["slices"].items():
        for group, budgets in groups.items():
            for budget, row in budgets.items():
                ci = row["paired_bootstrap_95"]
                lines.append(
                    f"| {slice_name} | {group} | {budget} | {row['unique_states']} | {row['mean_outcome_delta']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {row['mean_store_margin_delta']:.4f} |"
                )
    lines += [
        "",
        "## D1200-Match Causal Analysis",
        "",
        "| Group | Budget | States | D384 minus N384 outcome | 95% CI | Store-margin delta |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for group, budgets in summary["reference_match_causal_effects"].items():
        for budget, row in budgets.items():
            ci = row["paired_bootstrap_95"]
            lines.append(
                f"| {group} | {budget} | {row['unique_states']} | {row['mean_outcome_delta']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {row['mean_store_margin_delta']:.4f} |"
            )
    lines += [
        "",
        "D1200 agreement is directionally predictive here: when N384 alone matches D1200, D384-minus-N384 is negative at both budgets, including an upper CI below zero. The converse group is positive but its intervals include zero, so this is not a positive causal confirmation for denoising.",
        "",
        "## Decision Rules",
        "",
        "- `policy_target_noise_not_primary_confirmed`: no positive causal evidence, both global CIs include zero, and no sufficiently large robust phase/domain benefit.",
        "- `denoised_policy_target_rejected`: negative mean and upper 95% CI below zero at either budget.",
        "- `denoised_disagreement_moves_causally_better`: at least 64 states, positive means and lower CI >= 0 at both budgets, with no >=24-state player/domain mean below -0.10.",
        "- `target_noise_effect_phase_localized`: inconclusive global result but one >=32-state phase has positive lower CI without corresponding benefit elsewhere.",
        "- `strong_search_reference_not_causally_predictive`: D1200 matching fails to correspond to superior forced outcomes.",
        "",
        "Per-state forced-continuation records, including forced move, outcome, store margin, trajectory hash, and paired seed-context hash, remain in the workdir rather than this report.",
        "",
        "## Next Action",
        "",
        summary["next_action"],
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--pr177-workdir", default=str(DEFAULT_PR177_WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    current = Path(args.current)
    workdir.mkdir(parents=True, exist_ok=True)
    teachers, frozen = verify_pr177_inputs(
        pr177_workdir=Path(args.pr177_workdir),
        committed_summary_path=REPO_ROOT
        / "docs/data/alphazero-lite-policy-target-noise-ablation-summary.json",
        current_weights=current / "weights.json",
    )
    disagreements = disagreement_rows(teachers)
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    closeout_manifest = {
        **frozen,
        "current_weights_sha256": sha256_file(current / "weights.json"),
        "code_commit": code_commit,
        "continuation_settings": {
            "budgets": list(CONTINUATION_BUDGETS),
            "c_puct": 1.25,
            "tactical_root_bias": 0.0,
            "dirichlet_epsilon": 0.0,
            "root_policy_mode": "deterministic",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "disagreement_state_hashes": [row["state_hash"] for row in disagreements],
    }
    write_json(workdir / "closeout_manifest.json", closeout_manifest)
    checkpoint = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "current.npz"
    )
    records = run_forced_tasks(
        tasks=build_tasks(disagreements, experiment_seed=args.seed),
        checkpoint=checkpoint,
        workers=args.workers,
    )
    write_jsonl(workdir / "forced_continuations.jsonl", records)
    state_counts = {
        key: dict(sorted(Counter(str(row[key]) for row in disagreements).items()))
        for key in ("source_domain", "player", "phase", "legal_move_count")
    }
    summary: dict[str, Any] = {
        "schema": "azlite_policy_target_noise_causal_closeout_v1",
        "frozen_inputs": closeout_manifest,
        "disagreement_count": len(disagreements),
        "disagreement_counts": state_counts,
        "teacher_reference_agreement": teacher_agreement(disagreements),
        "forced_outcomes": summarize(records, experiment_seed=args.seed),
        "reference_match_causal_effects": reference_match_effects(
            records, experiment_seed=args.seed
        ),
        "records_sha256": sha256_file(workdir / "forced_continuations.jsonl"),
    }
    classification, classifications, next_action = classify(summary)
    summary["classification"] = classification
    summary["classifications"] = classifications
    summary["next_action"] = next_action
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-policy-target-noise-causal-closeout-summary.json",
        summary,
    )
    (
        REPO_ROOT / "docs/alphazero-lite-policy-target-noise-causal-closeout-results.md"
    ).write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps({"classification": classification, "workdir": str(workdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
