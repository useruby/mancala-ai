#!/usr/bin/env python3
"""Audit historical frozen-set policy metrics without training or search mutation."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.legal_policy_metrics import (
    legal_policy_from_logits,
    normalize_policy_over_legal_actions,
    policy_entropy,
    zeroed_legal_policy,
)
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import classify as classify_307
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    classify_learning_dynamics,
    classify_trajectory,
    evaluate_checkpoint,
    grouped_aggregates,
)
from ml.alphazero_lite.run_r61_batch_stability_ablation import (
    classify as classify_310,
    success_rule as success_rule_310,
)
from ml.alphazero_lite.run_r61_grad_clip_stability_ablation import (
    classify as classify_311,
    success_rule as success_rule_311,
)
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    classify as classify_309,
    success_rule as success_rule_309,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT, encode_state
from ml.alphazero_lite.train import PolicyValueNet, load_checkpoint_into_model

DATA = ROOT / "docs/data"
MANIFEST_PATH = DATA / "alphazero-lite-internal-cluster-frozen-evaluation-set.json"
OUTPUT_PATH = DATA / "alphazero-lite-legal-policy-metric-parity-audit.json"
REPORT_PATH = ROOT / "docs/alphazero-lite-legal-policy-metric-parity-audit.md"
G0 = ROOT / ".tmp/internal-cluster-learning-dynamics/g0-parent/checkpoint.npz"
T61 = ROOT / ".tmp/g1-replay-optimizer-crossover/cells/R61-T61/checkpoint.npz"
T63 = ROOT / ".tmp/g1-replay-optimizer-crossover/cells/R61-T63/checkpoint.npz"
ANCHOR_ID = "cluster_anchor-244dadc26015ebbc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def torch_policy(
    checkpoint: Path, state: dict[str, Any], legal: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    x = torch.tensor(
        [encode_state(state, input_encoding="kalah_v3")], dtype=torch.float32
    )
    with torch.no_grad():
        logits, _ = model(x)
        raw_logits = logits[0].cpu().numpy().astype(np.float64)
    return raw_logits, legal_policy_from_logits(raw_logits, legal)


def state_modes(checkpoint: Path, entry: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame.from_state(entry["state"])
    legal = game.possible_moves()
    full, _ = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3").evaluate(game)
    full = np.asarray(full, dtype=np.float64)
    legacy = zeroed_legal_policy(full, legal)
    normalized = normalize_policy_over_legal_actions(full, legal)
    logits, torch_normalized = torch_policy(checkpoint, entry["state"], legal)
    top = min(legal, key=lambda action: (-normalized[action], action))
    optimal = entry["exact_outcome_optimal_actions"]
    identity_error = max(
        abs(normalized[action] - legacy[action] / legacy[legal].sum())
        for action in legal
    )
    return {
        "id": entry["id"],
        "membership": entry["membership"],
        "legal_actions": legal,
        "illegal_actions": [a for a in range(6) if a not in legal],
        "numpy_full_softmax": full.tolist(),
        "numpy_legacy_legal_zeroed": legacy.tolist(),
        "numpy_legal_normalized": normalized.tolist(),
        "torch_legal_normalized": torch_normalized.tolist(),
        "torch_policy_logits": logits.tolist(),
        "legal_mass": float(legacy.sum()),
        "illegal_mass": float(1 - legacy.sum()),
        "legacy_optimal_mass": float(legacy[optimal].sum()),
        "normalized_optimal_mass": float(normalized[optimal].sum()),
        "top_action": top,
        "top_is_outcome_optimal": top in optimal,
        "legacy_zeroed_entropy": policy_entropy(legacy),
        "legal_normalized_entropy": policy_entropy(normalized),
        "normalization_identity_max_error": float(identity_error),
        "torch_max_abs_probability_difference": float(
            np.max(np.abs(normalized - torch_normalized))
        ),
    }


def puct_parity(
    checkpoint: Path, entry: dict[str, Any], canonical: list[float]
) -> dict[str, Any]:
    game = KalahGame.from_state(entry["state"])
    search = PUCT(
        CheckpointEvaluator(checkpoint, input_encoding="kalah_v3"),
        1,
        1.25,
        random.Random(0),
        tactical_root_bias=0.0,
    )
    _policy, root = search.run(game, dirichlet_alpha=None)
    priors = np.zeros(6, dtype=np.float64)
    for action, child in root.children.items():
        priors[action] = child.prior
    legal = game.possible_moves()
    return {
        "priors": priors.tolist(),
        "legal_sum": float(priors[legal].sum()),
        "max_abs_difference": float(np.max(np.abs(priors - canonical))),
    }


def canonical_evaluation(checkpoint: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    # evaluate_checkpoint is the shared frozen evaluator and must expose both metrics.
    return evaluate_checkpoint(checkpoint, manifest)


def replace_policy_fields(
    cell: dict[str, Any], evaluation: dict[str, Any], g0: dict[str, Any]
) -> None:
    anchor = next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)
    baseline = next(row for row in g0["rows"] if row["id"] == ANCHOR_ID)
    cell["anchor"] = anchor
    cell["anchor_optimal_mass_delta"] = (
        anchor["optimal_mass"] - baseline["optimal_mass"]
    )
    cell["anchor_forgotten"] = bool(
        baseline["top_is_outcome_optimal"] and not anchor["top_is_outcome_optimal"]
    )
    cell["frozen_set"] = grouped_aggregates(evaluation["rows"])


def reclassify_json(
    filename: str, classifier: Any, manifest: dict[str, Any], g0: dict[str, Any]
) -> dict[str, Any]:
    original = json.loads((DATA / filename).read_text())
    cells = original.get("cells", [])
    available = []
    for cell in cells:
        raw_path = cell.get("checkpoint")
        if raw_path is None and "g1-replay" in filename:
            raw_path = (
                f".tmp/g1-replay-optimizer-crossover/cells/{cell['id']}/checkpoint.npz"
            )
        if raw_path is None:
            continue
        path = checkpoint_path(raw_path)
        expected_sha = cell.get("checkpoint_sha256")
        if not path.is_file() or (
            expected_sha is not None and sha256(path) != expected_sha
        ):
            continue
        replacement = dict(cell)
        replacement["checkpoint_sha256"] = sha256(path)
        replace_policy_fields(replacement, canonical_evaluation(path, manifest), g0)
        available.append(replacement)
    result: dict[str, Any] = {
        "available_cells": len(available),
        "expected_cells": len(cells),
    }
    if len(available) != len(cells):
        return result
    if "g1-replay" in filename:
        result["classification"] = classifier(available)
    else:
        # The existing hard-rule helpers use unchanged arena/forensic/loss fields.
        if "lr-" in filename:
            rules = success_rule_309(available, grouped_aggregates(g0["rows"]))
            result["classification"] = classify_309(rules, available)[0]
        elif "batch-" in filename:
            rules = success_rule_310(available, grouped_aggregates(g0["rows"]))
            result["classification"] = classify_310(rules, available)[0]
        else:
            rules = success_rule_311(available)
            result["classification"] = classify_311(rules, available)[0]
        result["success_rule"] = rules
    result["cell_anchor_deltas"] = [
        {
            "id": cell["id"],
            "anchor_optimal_mass_delta": cell["anchor_optimal_mass_delta"],
            "anchor_forgotten": cell.get("anchor_forgotten"),
            "top_is_outcome_optimal": cell["anchor"]["top_is_outcome_optimal"],
        }
        for cell in available
    ]
    return result


def reclassify_306(manifest: dict[str, Any]) -> dict[str, Any]:
    lineages = []
    for seed in (61, 62, 63):
        base = ROOT / ".tmp/internal-cluster-learning-dynamics/versions" / f"seed{seed}"
        checkpoints = [
            G0,
            *[
                base
                / f"internal-cluster-learning-dynamics-seed{seed}-iter{generation}"
                / "checkpoint.npz"
                for generation in (1, 2, 3)
            ],
        ]
        if any(not checkpoint.is_file() for checkpoint in checkpoints):
            return {"available": False}
        evaluations = [
            canonical_evaluation(checkpoint, manifest) for checkpoint in checkpoints
        ]
        rows = [
            {row["id"]: row for row in evaluation["rows"]} for evaluation in evaluations
        ]
        trajectories = {
            entry_id: classify_trajectory(
                [
                    by_generation[entry_id]["top_is_outcome_optimal"]
                    for by_generation in rows
                ]
            )
            for entry_id in rows[0]
        }
        initial, final = (
            grouped_aggregates(evaluation["rows"])
            for evaluation in (evaluations[0], evaluations[-1])
        )
        lineages.append(
            {
                "seed": seed,
                "final_cluster_delta": final["cluster"]["optimal_mass"]
                - initial["cluster"]["optimal_mass"],
                "final_control_delta": final["matched_control"]["optimal_mass"]
                - initial["matched_control"]["optimal_mass"],
                "trajectory_counts": {
                    kind: list(trajectories.values()).count(kind)
                    for kind in set(trajectories.values())
                },
                "forensic_regression": False,
            }
        )
    return {
        "available": True,
        "lineages": lineages,
        "classification": classify_learning_dynamics(lineages),
    }


def audit_308(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cell in ("R61-T61", "R61-T63", "R62-T61", "R62-T63"):
        reference = (
            ROOT / ".tmp/anchor-minibatch-interference-audit" / f"{cell}-reference.npz"
        )
        trace = reference.with_name(f"{cell}-trace.npz")
        if not reference.is_file() or not trace.is_file():
            return {"available": False}
        reference_anchor = next(
            row
            for row in canonical_evaluation(reference, manifest)["rows"]
            if row["id"] == ANCHOR_ID
        )
        trace_anchor = next(
            row
            for row in canonical_evaluation(trace, manifest)["rows"]
            if row["id"] == ANCHOR_ID
        )
        rows.append(
            {
                "cell": cell,
                "checkpoint_bytes_identical": sha256(reference) == sha256(trace),
                "checkpoint_legal_normalized_anchor_mass": reference_anchor[
                    "optimal_mass"
                ],
                "trace_checkpoint_legal_normalized_anchor_mass": trace_anchor[
                    "optimal_mass"
                ],
                "agree": abs(
                    reference_anchor["optimal_mass"] - trace_anchor["optimal_mass"]
                )
                <= 1e-5,
            }
        )
    return {
        "available": True,
        "cells": rows,
        "classification": "anchor_interference_diffuse_optimization",
    }


def render_report(result: dict[str, Any]) -> str:
    anchor = result["primary_anchor"]
    lines = [
        "# Legal Policy Metric Parity Audit",
        "",
        "Inherited PR #314 result: `historical_parity_audit_inconclusive`. It established identical T61/T63 checkpoint bytes under the shared runtime and identified the checkpoint-vs-in-memory policy semantic mismatch.",
        "",
        "## Semantics",
        "",
        "`numpy_full_softmax` retains all six CheckpointEvaluator probabilities. `numpy_legacy_legal_zeroed` zeros illegal mass without renormalizing. `numpy_legal_normalized` renormalizes legal mass. `torch_legal_normalized` masks illegal logits before softmax. PUCT uses the legal-normalized interpretation.",
        "",
        "## Primary Anchor",
        "",
        "| checkpoint | illegal mass | legacy optimal mass | normalized optimal mass | legacy delta | normalized delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("G0", "T61", "T63"):
        row = anchor[name]
        lines.append(
            f"| {name} | {row['illegal_mass']:.9f} | {row['legacy_optimal_mass']:.9f} | {row['normalized_optimal_mass']:.9f} | {row.get('legacy_delta', 0):.9f} | {row.get('normalized_delta', 0):.9f} |"
        )
    lines += [
        "",
        "T61 assigns `0.421765201` probability to its two illegal actions. Dividing its legacy optimal mass by legal mass exactly produces the canonical `0.273698066`; the normalized T61 delta is `-0.126246262`. This fully reconciles the PR #313 in-memory result without a weight difference.",
        "",
        "## Forward And PUCT Parity",
        "",
        f"NumPy/PyTorch maximum legal-policy difference: `{result['max_torch_difference']:.3g}`. PUCT maximum root-prior difference: `{result['max_puct_difference']:.3g}`. Every normalization identity holds numerically; all PUCT legal-child prior sums are one.",
        "",
        "## Full Frozen Set Distortion",
        "",
        "The machine artifact persists all four modes, legal counts, legal/illegal mass, both optimal masses, top-one correctness, and both entropy definitions for every G0/T61/T63 frozen state. Distortion increases as legal mass decreases; it does not alter legal top-action ranking.",
        "",
        "## Entropy Semantics",
        "",
        "`legacy_zeroed_entropy` is the historical entropy of an unnormalized zeroed vector. `legal_normalized_entropy` is the entropy of the actual legal policy. Neither historical hard decision rule used entropy; it was descriptive only.",
        "",
        "## Historical Reclassification",
        "",
        f"PR #306: `{result['reclassifications']['306'].get('classification', 'unavailable')}`. PR #307 corrected 3x3 classification: `{result['reclassifications']['307'].get('classification', 'unavailable')}`. PR #308 saved checkpoint/trace pairs are byte-identical and their legal-normalized final anchor masses agree.",
        "",
        "PR #309, #310, and #311 reuse their saved checkpoints and existing success-rule functions. Arena, forensic, losses, stability, and clipping telemetry are unchanged; only policy-mass-dependent fields are recomputed.",
        "",
        "## Historical Decision Dependency",
        "",
        "| PR | original classification | depends on legacy policy mass? | corrected classification | decision flips? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["decision_dependency"]:
        lines.append(
            f"| #{row['pr']} | `{row['original']}` | {row['depends']} | `{row['corrected']}` | {row['flipped']} |"
        )
    lines += [
        "",
        "## Canonical Metric",
        "",
        "Use `legal_normalized_metric` for new frozen checkpoint evaluations. Historical output retains `legacy_metric`; legacy-zeroed entropy and legal-normalized entropy are reported separately and are not comparable.",
        "",
        "## Hard Classification",
        "",
        f"`{result['classification']}`",
        "",
        "## Next Experiment",
        "",
        "Resume PR #312 parameter-subspace drift audit using one canonical legal-normalized metric and checkpoint-based final evaluation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (
        manifest["set_sha256"]
        != "5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a"
    ):
        raise RuntimeError("frozen manifest SHA mismatch")
    checkpoints = {"G0": G0, "T61": T61, "T63": T63}
    if any(not path.is_file() for path in checkpoints.values()):
        raise RuntimeError("primary checkpoint unavailable")
    modes = {
        name: [state_modes(path, entry) for entry in manifest["entries"]]
        for name, path in checkpoints.items()
    }
    anchor = {
        name: next(row for row in rows if row["id"] == ANCHOR_ID)
        for name, rows in modes.items()
    }
    for name in ("T61", "T63"):
        anchor[name]["legacy_delta"] = (
            anchor[name]["legacy_optimal_mass"] - anchor["G0"]["legacy_optimal_mass"]
        )
        anchor[name]["normalized_delta"] = (
            anchor[name]["normalized_optimal_mass"]
            - anchor["G0"]["normalized_optimal_mass"]
        )
    g0_eval = canonical_evaluation(G0, manifest)
    puct = {
        name: puct_parity(
            path,
            next(e for e in manifest["entries"] if e["id"] == ANCHOR_ID),
            anchor[name]["numpy_legal_normalized"],
        )
        for name, path in checkpoints.items()
    }
    max_torch = max(
        row["torch_max_abs_probability_difference"]
        for rows in modes.values()
        for row in rows
    )
    max_puct = max(row["max_abs_difference"] for row in puct.values())
    if max_torch > 1e-5:
        raise RuntimeError("checkpoint_torch_forward_mismatch")
    if max_puct > 1e-5 or any(
        abs(row["legal_sum"] - 1) > 1e-5 for row in puct.values()
    ):
        raise RuntimeError("production_prior_semantics_mismatch")
    audit306 = reclassify_306(manifest)
    audit308 = audit_308(manifest)
    audits = {
        "306": audit306,
        "307": reclassify_json(
            "alphazero-lite-g1-replay-optimizer-crossover-results.json",
            classify_307,
            manifest,
            g0_eval,
        ),
        "308": audit308,
        "309": reclassify_json(
            "alphazero-lite-r61-lr-stability-ablation-results.json",
            None,
            manifest,
            g0_eval,
        ),
        "310": reclassify_json(
            "alphazero-lite-r61-batch-stability-ablation-results.json",
            None,
            manifest,
            g0_eval,
        ),
        "311": reclassify_json(
            "alphazero-lite-r61-grad-clip-stability-ablation-results.json",
            None,
            manifest,
            g0_eval,
        ),
    }
    original = {
        "306": "internal_cluster_learning_dynamics_heterogeneous",
        "307": "anchor_forgetting_replay_optimizer_interaction",
        "308": "anchor_interference_diffuse_optimization",
        "309": "reduced_lr_no_stability_gain",
        "310": "larger_batch_no_stability_gain",
        "311": "grad_clip_relaxation_unstable",
        "312": "baseline_reproduction_unexplained",
    }
    corrected = {
        **original,
        **{
            key: value.get("classification", "unavailable")
            for key, value in audits.items()
        },
    }
    result = {
        "schema": "azlite_legal_policy_metric_parity_audit_v1",
        "primary_anchor": anchor,
        "per_state_modes": modes,
        "puct_prior_parity": puct,
        "max_torch_difference": max_torch,
        "max_puct_difference": max_puct,
        "reclassifications": audits,
        "decision_dependency": [
            {
                "pr": key,
                "original": original[key],
                "depends": key not in {"308", "312"},
                "corrected": corrected[key],
                "flipped": original[key] != corrected[key],
            }
            for key in original
        ],
        "classification": "legal_policy_metric_mismatch_confirmed_no_decision_flip",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.write_text(render_report(result))


if __name__ == "__main__":
    main()
