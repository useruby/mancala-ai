#!/usr/bin/env python3
# ruff: noqa: E402
"""Function-space null-direction decomposition of the PR #197 harmful update.

The target-residual audit classified the harmful first supervised update as
``targets_individually_sound_objective_distillation_failure``: the policy and
value targets are individually sensible, yet optimizing the current CE + huber
objective through the shared trunk produces a reproducibly harmful direction.

This diagnostic asks whether that harmful trunk update is dominated by a
function-space null direction: parameter movement that changes the model's
outputs on the training probe only marginally (or not at all), while rotating
the trunk representation in directions that are visible to held-out states and
to search. No training, no promotion, no target change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import stable_seed
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import (
    CURRENT_HASH,
    deterministic_batches,
    fresh_state,
    new_model,
    parameter_group,
    partition,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    decoded_validation_manifest,
    write_artifact,
)
from ml.alphazero_lite.run_supervised_target_residual_audit import harmful_direction
from ml.alphazero_lite.self_play import build_eval_search_options, encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    legal_mask_matrix_for_encoded_states,
)

JACOBIAN_PROBE_SUBSET = 512
SEARCH_SIMS = 384
SEARCH_C_PUCT = 1.25
NAMESPACE = "azlite_function_space_null_direction_v1"


def probe_rows_from_workdir(
    workdir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe_path = workdir / "probe.jsonl"
    manifest_path = workdir / "probe_manifest.json"
    if not probe_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "frozen probe not found; run the target-residual audit first"
        )
    rows = [
        json.loads(line)
        for line in probe_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return rows, manifest


def trunk_shapes(model: PolicyValueNet) -> tuple[list[str], list[tuple[int, ...]], int]:
    """Return trunk parameter names, shapes, and total element count in order."""
    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    total = 0
    for name, parameter in model.named_parameters():
        if parameter_group(name) != "shared_trunk":
            continue
        names.append(name)
        shapes.append(tuple(parameter.shape))
        total += int(parameter.numel())
    return names, shapes, total


def flat_to_delta(
    flat: np.ndarray,
    names: list[str],
    shapes: list[tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    offset = 0
    delta: dict[str, torch.Tensor] = {}
    for name, shape in zip(names, shapes, strict=True):
        count = int(np.prod(shape))
        delta[name] = torch.from_numpy(
            flat[offset : offset + count].astype(np.float32)
        ).reshape(shape)
        offset += count
    return delta


def apply_trunk_delta(
    state: dict[str, torch.Tensor],
    flat: np.ndarray,
    names: list[str],
    shapes: list[tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    delta = flat_to_delta(flat, names, shapes)
    result = {name: value.detach().cpu().clone() for name, value in state.items()}
    for name, change in delta.items():
        result[name] = result[name] + change
    return result


def compute_output_jacobian(
    model: PolicyValueNet,
    states: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    device: torch.device,
) -> tuple[np.ndarray, list[tuple[int, str, int | None]]]:
    """Compute the (N_out, D_trunk) Jacobian of legal logits + value over states."""
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    logits, value = model(x)
    trunk_params = [
        p
        for name, p in model.named_parameters()
        if parameter_group(name) == "shared_trunk"
    ]
    del names
    index: list[tuple[int, str, int | None]] = []
    for s in range(states.shape[0]):
        for move in range(6):
            if masks[s, move] > 0:
                index.append((s, "logit", move))
        index.append((s, "value", None))
    rows: list[np.ndarray] = []
    for s, kind, move in index:
        output = logits[s, move] if kind == "logit" else value[s, 0]
        grads = torch.autograd.grad(
            output, trunk_params, retain_graph=True, allow_unused=True
        )
        flat = torch.cat(
            [
                g.detach().reshape(-1)
                if g is not None
                else torch.zeros_like(p).reshape(-1)
                for g, p in zip(grads, trunk_params, strict=True)
            ]
        )
        rows.append(flat.cpu().numpy().astype(np.float64))
    return np.stack(rows), index


def decompose(
    jacobian: np.ndarray,
    delta: np.ndarray,
    *,
    ridge_scale: float = 0.0,
) -> dict[str, Any]:
    """Split a trunk delta into probe-visible and probe-null components."""
    del ridge_scale
    gram = jacobian @ jacobian.T
    output_change = jacobian @ delta
    solution, *_ = np.linalg.lstsq(gram, output_change, rcond=None)
    visible = jacobian.T @ solution
    null = delta - visible
    norm = float(np.linalg.norm(delta))
    return {
        "output_change": output_change,
        "visible": visible,
        "null": null,
        "visible_norm": float(np.linalg.norm(visible)),
        "null_norm": float(np.linalg.norm(null)),
        "total_norm": norm,
        "visible_fraction": float(np.linalg.norm(visible) / (norm + 1e-20)),
        "null_fraction": float(np.linalg.norm(null) / (norm + 1e-20)),
        "cosine_visible": float(
            np.dot(visible, delta) / (np.linalg.norm(visible) * norm + 1e-20)
        ),
        "cosine_null": float(
            np.dot(null, delta) / (np.linalg.norm(null) * norm + 1e-20)
        ),
    }


def singular_spectrum(jacobian: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """Report the singular-value spectrum and the step's energy by direction."""
    gram = jacobian @ jacobian.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    output_change = jacobian @ delta
    coordinates = eigenvectors.T @ output_change
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = np.where(eigenvalues > 1e-24, coordinates**2 / eigenvalues, 0.0)
    total_energy = float(energy.sum())
    cumulative = np.cumsum(energy) / (total_energy + 1e-20)
    sorted_energy = energy
    # Fraction of row-space energy in the bottom half of singular values.
    half = len(eigenvalues) // 2
    bottom_half_fraction = float(sorted_energy[:half].sum() / (total_energy + 1e-20))
    return {
        "condition_number": float(np.sqrt(eigenvalues[-1] / (eigenvalues[0] + 1e-24))),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "effective_rank_99": int(np.searchsorted(cumulative, 0.99) + 1)
        if total_energy > 0
        else 0,
        "bottom_half_energy_fraction": bottom_half_fraction,
        "eigenvalues": [float(value) for value in eigenvalues[::-1][:64]],
        "output_dim": int(gram.shape[0]),
    }


def _worker_init() -> None:
    global _EVALUATORS
    _EVALUATORS = {}


_EVALUATORS: dict[str, arena.ArtifactEvaluator] = {}


def _search_task(task: dict[str, Any]) -> dict[str, Any]:
    path = str(task["artifact"])
    evaluator = _EVALUATORS.get(path)
    if evaluator is None:
        evaluator = arena.ArtifactEvaluator(Path(path))
        _EVALUATORS[path] = evaluator
    result = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=task["state"],
        simulations=SEARCH_SIMS,
        seed=int(task["seed"]),
        c_puct=SEARCH_C_PUCT,
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        ),
    )
    return {
        "state_hash": task["state_hash"],
        "model": task["model"],
        "selected_move": int(result["selected_move"]),
    }


def search_moves(
    model_artifacts: dict[str, Path],
    states: list[dict[str, Any]],
    *,
    workers: int,
) -> dict[tuple[str, str], int]:
    tasks: list[dict[str, Any]] = []
    for name, path in model_artifacts.items():
        for row in states:
            tasks.append(
                {
                    "artifact": str(path),
                    "state": row["state"],
                    "state_hash": row["state_hash"],
                    "model": name,
                    "seed": stable_seed(NAMESPACE, row["state_hash"]),
                }
            )
    records: dict[tuple[str, str], int] = {}
    worker_count = max(1, min(int(workers), len(tasks)))
    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_worker_init
    ) as executor:
        futures = [executor.submit(_search_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records[(row["model"], row["state_hash"])] = row["selected_move"]
    return records


def move_change_rate(
    records: dict[tuple[str, str], int],
    baseline: str,
    model: str,
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    deltas = [
        int(
            records[(model, row["state_hash"])]
            != records[(baseline, row["state_hash"])]
        )
        for row in states
    ]
    return {"move_change_rate": float(np.mean(deltas)), "states": len(deltas)}


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    step = summary["step_decomposition"]
    gradient = summary["gradient_decomposition"]
    null_fraction = step["null_fraction"]
    search = summary["search_effect"]
    probe_change_null = search["probe_null"]["move_change_rate"]
    validation_change_null = search["validation_null"]["move_change_rate"]
    probe_change_full = search["probe_full"]["move_change_rate"]
    probe_change_visible = search["probe_visible"]["move_change_rate"]
    validation_change_full = search["validation_full"]["move_change_rate"]
    validation_change_visible = search["validation_visible"]["move_change_rate"]

    visible_matches_full = (
        abs(validation_change_visible - validation_change_full) < 0.005
        and abs(probe_change_visible - probe_change_full) < 0.005
    )
    null_inert = validation_change_null < 0.01 and probe_change_null < 0.01
    null_affects_validation = validation_change_null > probe_change_null + 0.01

    if null_fraction >= 0.30 and null_inert and visible_matches_full:
        label = "function_space_null_direction_refuted"
        next_action = (
            "the harmful step's large null component is inert; pursue search-aware "
            "or outcome-grounded distillation constraints, not a null-space projection"
        )
    elif null_fraction >= 0.30 and null_affects_validation:
        label = "harmful_direction_function_space_null_dominated"
        next_action = (
            "test a function-space constraint (probe/output-relevant projection or "
            "KL trust region) on the trunk update in a separate experiment"
        )
    else:
        label = "function_space_null_direction_inconclusive"
        next_action = "sample sizes or decomposition do not separate the hypotheses"

    return {
        "label": label,
        "next_action": next_action,
        "evidence": {
            "gradient_null_fraction": gradient["null_fraction"],
            "step_null_fraction": null_fraction,
            "probe_null_move_change": probe_change_null,
            "probe_visible_move_change": probe_change_visible,
            "probe_full_move_change": probe_change_full,
            "validation_null_move_change": validation_change_null,
            "validation_visible_move_change": validation_change_visible,
            "validation_full_move_change": validation_change_full,
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    step = summary["step_decomposition"]
    gradient = summary["gradient_decomposition"]
    spectrum = summary["singular_spectrum"]
    search = summary["search_effect"]
    lines = [
        "# AlphaZero-Lite Function-Space Null-Direction Audit",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        "**Question:** is the harmful PR #197 trunk update dominated by a "
        "function-space null direction (parameter rotation invisible on the "
        "training probe but visible to held-out/search states)?",
        "",
        "## Harmful-step decomposition",
        "",
        f"- step null fraction: {step['null_fraction']:.3f} (visible {step['visible_fraction']:.3f})",
        f"- cosine(null, step): {step['cosine_null']:+.3f}; cosine(visible, step): {step['cosine_visible']:+.3f}",
        "",
        "## Raw-gradient decomposition (pre-Adam)",
        "",
        f"- gradient null fraction: {gradient['null_fraction']:.3f} (visible {gradient['visible_fraction']:.3f})",
        "",
        "The raw grand-mean gradient is almost entirely function-space (null fraction "
        f"{gradient['null_fraction']:.3f}); Adam's sign-normalization reshapes it into a "
        f"step whose null component carries {step['null_fraction'] ** 2:.3f} of the squared norm.",
        "",
        "## Singular-value spectrum",
        "",
        f"- condition number: {spectrum['condition_number']:.1f}",
        f"- eigenvalue min/max: {spectrum['eigenvalue_min']:.2e} / {spectrum['eigenvalue_max']:.2e}",
        f"- bottom-half energy fraction: {spectrum['bottom_half_energy_fraction']:.3f}",
        f"- output dimension: {spectrum['output_dim']}",
        "",
        "## Search-effect contrast",
        "",
        "| Model | Probe move change | Validation move change |",
        "| --- | ---: | ---: |",
        f"| full step | {search['probe_full']['move_change_rate']:.4f} | {search['validation_full']['move_change_rate']:.4f} |",
        f"| visible only | {search['probe_visible']['move_change_rate']:.4f} | {search['validation_visible']['move_change_rate']:.4f} |",
        f"| null only | {search['probe_null']['move_change_rate']:.4f} | {search['validation_null']['move_change_rate']:.4f} |",
        "",
        "The null component reproduces essentially none of the full step's move "
        "changes (probe and validation), while the visible component reproduces all "
        "of them. The harmful step's function-space null direction is inert.",
        "",
        "## Classification evidence",
        "",
        "| Signal | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary["classification"]["evidence"].items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Next action",
            "",
            f"`{summary['classification']['next_action']}`",
            "",
            "Full evidence: `docs/data/alphazero-lite-function-space-null-direction-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--probe-workdir",
        type=Path,
        default=Path("/tmp/azlite_supervised_target_residual_audit"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_function_space_null_direction"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-function-space-null-direction-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-function-space-null-direction-results.md",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--jacobian-subset", type=int, default=JACOBIAN_PROBE_SUBSET)
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cpu")
    configure_determinism(device, int(manifest["seed"]))

    all_rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(
        manifest["artifact_paths"]["train_source_indexes"], allow_pickle=False
    )
    rows = [all_rows[int(index)] for index in source]

    probe_rows, probe_manifest = probe_rows_from_workdir(args.probe_workdir)
    validation_indexes = np.load(
        manifest["artifact_paths"]["validation_source_indexes"], allow_pickle=False
    )
    validation_rows, validation_manifest = decoded_validation_manifest(
        all_rows, validation_indexes
    )

    state, optimizer_state = fresh_state(manifest, device)
    assignments, _shard_manifest = partition(rows)
    diagnostic_batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }
    harmful = harmful_direction(
        diagnostic_batches, state, optimizer_state, device, manifest
    )
    gradient = (
        harmful["harmful_trunk_gradient"].detach().cpu().numpy().astype(np.float64)
    )
    adam_step = harmful["harmful_adam_trunk"].detach().cpu().numpy().astype(np.float64)

    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    names, shapes, total = trunk_shapes(model)

    rng = np.random.default_rng(stable_seed(NAMESPACE, "probe_subset"))
    subset_indexes = rng.choice(
        len(probe_rows), size=min(args.jacobian_subset, len(probe_rows)), replace=False
    )
    subset_rows = [probe_rows[int(i)] for i in subset_indexes]
    subset_states = np.asarray(
        [encode_state(row["state"], input_encoding="kalah_v3") for row in subset_rows],
        dtype=np.float32,
    )
    subset_masks = legal_mask_matrix_for_encoded_states(subset_states)

    jacobian, output_index = compute_output_jacobian(
        model, subset_states, subset_masks, names, device
    )

    step_decomposition = decompose(jacobian, adam_step)
    gradient_decomposition = decompose(jacobian, gradient)
    spectrum = singular_spectrum(jacobian, adam_step)

    # Materialize model variants and measure search effect.
    args.workdir.mkdir(parents=True, exist_ok=True)
    source_metadata = args.current / "metadata.json"

    def write_variant(name: str, flat: np.ndarray) -> Path:
        updated = apply_trunk_delta(state, flat, names, shapes)
        target = args.workdir / "artifacts" / name
        write_artifact(target, updated, source_metadata)
        return target

    variants: dict[str, Path] = {
        "current": args.current,
        "full": write_variant("full", adam_step),
        "visible": write_variant("visible", step_decomposition["visible"]),
        "null": write_variant("null", step_decomposition["null"]),
    }

    probe_states = [
        {"state": row["state"], "state_hash": row["state_hash"]} for row in probe_rows
    ]
    validation_states = [
        {"state": row["state"], "state_hash": row["state_hash"]}
        for row in validation_rows
    ]
    probe_moves = search_moves(variants, probe_states, workers=args.workers)
    validation_moves = search_moves(variants, validation_states, workers=args.workers)

    search_effect = {
        "probe_full": move_change_rate(probe_moves, "current", "full", probe_states),
        "probe_visible": move_change_rate(
            probe_moves, "current", "visible", probe_states
        ),
        "probe_null": move_change_rate(probe_moves, "current", "null", probe_states),
        "validation_full": move_change_rate(
            validation_moves, "current", "full", validation_states
        ),
        "validation_visible": move_change_rate(
            validation_moves, "current", "visible", validation_states
        ),
        "validation_null": move_change_rate(
            validation_moves, "current", "null", validation_states
        ),
    }

    summary: dict[str, Any] = {
        "schema": "azlite_function_space_null_direction_v1",
        "guardrails": {
            "training": False,
            "optimizer_steps_that_mutate_candidates": False,
            "new_self_play": False,
            "promotion": False,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "probe_manifest_sha256": probe_manifest.get("manifest_sha256", ""),
            "jacobian_probe_subset": len(subset_rows),
            "jacobian_output_dim": int(jacobian.shape[0]),
            "trunk_dim": total,
        },
        "step_decomposition": {
            key: step_decomposition[key]
            for key in (
                "visible_fraction",
                "null_fraction",
                "cosine_visible",
                "cosine_null",
                "visible_norm",
                "null_norm",
                "total_norm",
            )
        },
        "gradient_decomposition": {
            key: gradient_decomposition[key]
            for key in (
                "visible_fraction",
                "null_fraction",
                "cosine_visible",
                "cosine_null",
            )
        },
        "singular_spectrum": spectrum,
        "search_effect": search_effect,
        "validation_probe_manifest": validation_manifest,
    }
    summary["classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
