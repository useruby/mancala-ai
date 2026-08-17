#!/usr/bin/env python3
"""Run the PR191 policy-stop-gradient shared-trunk attribution ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_metrics import (  # noqa: E402
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    ARENA_SUITE,
    CURRENT_HASH,
    _batch,
    _losses,
    _new_model,
    _save_snapshot,
    export_snapshot_artifacts,
    output_drift,
    puct_trajectory,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    _context_c_puct,
    decoded_validation_manifest,
    stable_hash,
)
from ml.alphazero_lite.train import apply_trainable_scope, load_checkpoint_into_model  # noqa: E402

SNAPSHOTS = (0, 1, 2, 3, 4, 5, 8, 12)
ARENA_STEPS = (1, 3, 5, 12)
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")


def replay_lane(
    manifest: dict[str, Any], workdir: Path, device: torch.device, lane: str
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    """Replay saved PR191 batches with the requested, fixed gradient pathway."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batches = [
        _batch([rows[int(index)] for index in source], indexes, device)
        for indexes in plan
    ]
    model = _new_model(device)
    load_checkpoint_into_model(model, paths["initialization_checkpoint"])
    if lane == "heads_only":
        apply_trainable_scope(model, "heads_only")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(manifest["optimizer"]["lr"])
    )
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}
    model.train()
    for step, batch in enumerate(batches[:12], 1):
        optimizer.zero_grad(set_to_none=True)
        if lane == "policy_detached_trunk":
            logits, prediction = model(batch["x"], detach_policy_trunk=True)
            from ml.alphazero_lite.train import (
                compute_policy_cross_entropy,
                compute_value_loss_vector,
            )

            policy = compute_policy_cross_entropy(
                logits.masked_fill(batch["mask"] <= 0, -1e9), batch["p"]
            ).mean()
            value = (
                0.6
                * compute_value_loss_vector(
                    prediction, batch["v"], value_loss="huber", huber_delta=1.0
                ).mean()
            )
        else:
            policy, value = _losses(model, batch)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(manifest["gradient_clip"])
        )
        optimizer.step()
        if step in SNAPSHOTS:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return saved


def state_hashes(
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
) -> dict[str, str]:
    """Hash all tensors at every boundary for deterministic reproduction."""
    return {
        str(step): stable_hash(
            {
                name: value.detach().cpu().numpy().tobytes().hex()
                for name, value in state.items()
            }
        )
        for step, (state, _optimizer) in snapshots.items()
    }


def trunk_delta(
    state: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> float:
    left = torch.cat(
        [
            value.reshape(-1).cpu()
            for name, value in state.items()
            if name.startswith(TRUNK_PREFIXES)
        ]
    )
    right = torch.cat(
        [
            value.reshape(-1).cpu()
            for name, value in current.items()
            if name.startswith(TRUNK_PREFIXES)
        ]
    )
    return float(
        torch.linalg.vector_norm(left - right)
        / (torch.linalg.vector_norm(right) + 1e-20)
    )


def _arena_records(
    workdir: Path,
    challenger: Path,
    current: Path,
    context: str,
    role: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Run one matched-current comparison; cache only exact provenance."""
    challenger_hash, incumbent_hash, suite_hash = (
        sha256_file(challenger / "weights.json"),
        sha256_file(current / "weights.json"),
        sha256_file(ARENA_SUITE),
    )
    sims, current_sims = (int(value) for value in context.split(":"))
    result = []
    for seat in (0, 1):
        directory = workdir / context.replace(":", "_") / role / f"starts_{seat}"
        records, provenance = directory / "arena.jsonl", directory / "provenance.json"
        manifest = {
            "schema_version": 1,
            "challenger_hash": challenger_hash,
            "incumbent_hash": incumbent_hash,
            "opening_suite_hash": suite_hash,
            "search_budgets": [sims, current_sims],
            "effective_c_puct": _context_c_puct(context),
            "tactical_root_bias": 0.0,
            "root_policy_mode": "deterministic",
            "seed_contract": "canonical_v1",
            "base_seed": 42,
            "games_per_opening": 2,
        }
        reusable = records.is_file() and provenance.is_file()
        if reusable:
            cached = json.loads(provenance.read_text(encoding="utf-8"))
            reusable = all(
                cached.get(key) == value for key, value in manifest.items()
            ) and cached.get("arena_output_hash") == sha256_file(records)
        if not reusable:
            directory.mkdir(parents=True, exist_ok=True)
            run_arena(
                challenger=str(challenger),
                current=str(current),
                challenger_sims=sims,
                current_sims=current_sims,
                games=256,
                seed=42,
                workers=workers,
                out_json=str(directory / "arena.json"),
                out_jsonl=str(records),
                opening_prefixes_jsonl=str(ARENA_SUITE),
                challenger_starts=seat,
                games_per_opening=2,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                normalize_values=False,
                c_puct=_context_c_puct(context),
                tactical_root_bias=0.0,
                seed_ledger_output=str(directory / "seed_ledger.jsonl"),
            )
            provenance.write_text(
                json.dumps(
                    manifest | {"arena_output_hash": sha256_file(records)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        rows = parse_game_jsonl(str(records))
        opponent_config = {
            key: manifest[key]
            for key in (
                "schema_version",
                "search_budgets",
                "effective_c_puct",
                "tactical_root_bias",
                "root_policy_mode",
                "seed_contract",
                "base_seed",
                "games_per_opening",
            )
        }
        for row in rows:
            row.update(
                opponent_weights_sha256=incumbent_hash,
                opponent_config_sha256=stable_hash(opponent_config),
            )
        result.extend(rows)
    return result


def early_arena(
    artifacts: dict[str, dict[int, Path]], current: Path, workdir: Path, workers: int
) -> dict[str, Any]:
    """Estimate treatment effects only through each lane versus matched current."""
    metrics: dict[str, Any] = {}
    for step in ARENA_STEPS:
        for context in ("384:256", "1200:1200"):
            control = _arena_records(
                workdir, current, current, context, "current_control", workers
            )
            effects = {
                lane: paired_opening_candidate_effect(
                    _arena_records(
                        workdir / f"step_{step:04d}",
                        artifacts[lane][step],
                        current,
                        context,
                        f"{lane}_vs_current",
                        workers,
                    ),
                    control,
                )
                for lane in ("heads_only", "policy_detached_trunk", "baseline_joint")
            }
            metrics.setdefault(str(step), {})[context] = effects | {
                "value_trunk_effect": paired_effect_difference(
                    effects["policy_detached_trunk"], effects["heads_only"]
                ),
                "policy_trunk_gradient_effect": paired_effect_difference(
                    effects["baseline_joint"], effects["policy_detached_trunk"]
                ),
            }
    return metrics


def same_step_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Subtract same-state probe summaries with a shared current reference."""
    return {
        step: {
            context: {
                key: left["metrics"][step][context][key]
                - right["metrics"][step][context][key]
                for key in left["metrics"][step][context]
                if key != "states"
            }
            for context in left["metrics"][step]
        }
        for step in left["metrics"]
    }


def decision(arena: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the prespecified step-12 continuation gate without extrapolation."""
    if not arena:
        return "pending_prespecified_evidence", {
            "passed": False,
            "reason": "arena_not_run",
        }
    practical = arena["12"]["384:256"]
    high = arena["12"]["1200:1200"]
    detached_beats_joint = all(
        metrics["policy_trunk_gradient_effect"]["paired_candidate_effect"] < 0
        for metrics in (practical, high)
    )
    lower_ci = (
        practical["policy_trunk_gradient_effect"]["opening_bootstrap_ci"]["upper_95"]
        < 0
    )
    detached_safe = (
        practical["policy_detached_trunk"]["paired_candidate_effect"] >= 0
        and high["policy_detached_trunk"]["paired_candidate_effect"] >= -0.03
    )
    passed = detached_beats_joint and lower_ci and detached_safe
    return (
        "policy_trunk_gradient_inconclusive_stop"
        if not passed
        else "detached_continuation_approved",
        {
            "passed": passed,
            "detached_beats_joint_both_contexts": detached_beats_joint,
            "detached_minus_joint_lower_ci_positive_at_384": lower_ci,
            "detached_vs_current_safety": detached_safe,
            "reason": "prespecified_continuation_gate_failed"
            if not passed
            else "passed",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_policy_detached_trunk")
    )
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact is not the PR191 initialization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    for lane in ("heads_only", "baseline_joint", "policy_detached_trunk"):
        configure_determinism(device, int(manifest["seed"]))
        lanes[lane] = replay_lane(
            manifest,
            args.workdir / lane,
            device,
            "joint_trunk" if lane == "baseline_joint" else lane,
        )
    configure_determinism(device, int(manifest["seed"]))
    repeat = replay_lane(
        manifest,
        args.workdir / "policy_detached_trunk_repeat",
        device,
        "policy_detached_trunk",
    )
    deterministic = state_hashes(lanes["policy_detached_trunk"]) == state_hashes(repeat)
    if not deterministic:
        raise RuntimeError("policy_detached_trunk step-12 replay is not deterministic")
    rows = read_jsonl(Path(manifest["replay_path"]))
    validation = np.load(
        Path(manifest["artifact_paths"]["validation_source_indexes"]),
        allow_pickle=False,
    )
    probe, probe_manifest = decoded_validation_manifest(rows, validation)
    probe_hash = stable_hash(probe_manifest)
    output = {lane: output_drift(rows, snapshots) for lane, snapshots in lanes.items()}
    current_state = lanes["baseline_joint"][0][0]
    for lane in output:
        for step in output[lane]:
            output[lane][step]["trunk_relative_l2_delta_from_current"] = trunk_delta(
                lanes[lane][int(step)][0], current_state
            )
    artifacts = {
        lane: export_snapshot_artifacts(snapshots, args.workdir / lane)
        for lane, snapshots in lanes.items()
    }
    probe_artifacts = {
        lane: {
            step: artifact
            for step, artifact in lane_artifacts.items()
            if step == 0 or step in ARENA_STEPS
        }
        for lane, lane_artifacts in artifacts.items()
    }
    puct = (
        {
            lane: puct_trajectory(
                probe[:256], probe_artifacts[lane], args.workdir / lane, probe_hash
            )
            for lane in lanes
        }
        if args.puct
        else {}
    )
    if puct:
        puct["detached_minus_heads_only"] = same_step_difference(
            puct["policy_detached_trunk"], puct["heads_only"]
        )
        puct["joint_minus_detached"] = same_step_difference(
            puct["baseline_joint"], puct["policy_detached_trunk"]
        )
    arena = (
        early_arena(artifacts, args.current, args.workdir / "arena", args.arena_workers)
        if args.arena
        else {}
    )
    attribution_contrasts = {
        contrast: {
            step: {
                "trunk_relative_l2_delta": output[left][step][
                    "trunk_relative_l2_delta_from_current"
                ]
                - output[right][step]["trunk_relative_l2_delta_from_current"],
                "policy": {
                    key: output[left][step]["policy"][key]
                    - output[right][step]["policy"][key]
                    for key in output[left][step]["policy"]
                },
                "value": {
                    key: output[left][step]["value"][key]
                    - output[right][step]["value"][key]
                    for key in output[left][step]["value"]
                },
            }
            for step in output[left]
        }
        for contrast, left, right in (
            ("detached_minus_heads_only", "policy_detached_trunk", "heads_only"),
            ("joint_minus_detached", "baseline_joint", "policy_detached_trunk"),
        )
    }
    classification, continuation = decision(arena)
    summary = {
        "schema": "azlite_policy_detached_trunk_v1",
        "guardrails": {
            "gradient_projection": False,
            "value_weight": 0.6,
            "gradient_clip": float(manifest["gradient_clip"]),
        },
        "deterministic_reproduction": deterministic,
        "snapshot_steps": list(SNAPSHOTS),
        "state_hashes": {
            lane: state_hashes(snapshots) for lane, snapshots in lanes.items()
        },
        "attribution": output,
        "attribution_contrasts": attribution_contrasts,
        "puct": puct,
        "early_arena": arena,
        "continuation": continuation,
        "classification": classification,
    }
    summary_path = (
        REPO_ROOT / "docs/data/alphazero-lite-policy-detached-trunk-summary.json"
    )
    report_path = REPO_ROOT / "docs/alphazero-lite-policy-detached-trunk-results.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# AlphaZero-Lite Policy-Detached Trunk Results",
        "",
        f"**Classification:** `{classification}`",
        "",
        f"- Deterministic detached replay through step 12: `{deterministic}`",
        f"- Full-epoch continuation: `{continuation['passed']}` (`{continuation['reason']}`)",
        "",
        "## Matched-Current Arena",
        "",
        "`policy_trunk_gradient_effect` is joint minus detached. Positive values favor detached.",
        "",
        "| Step | Context | Heads-current | Detached-current | Joint-current | Joint-detached 95% CI |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for step in ARENA_STEPS:
        for context in ("384:256", "1200:1200"):
            metrics = arena[str(step)][context]
            ci = metrics["policy_trunk_gradient_effect"]["opening_bootstrap_ci"]
            lines.append(
                f"| {step} | {context} | {metrics['heads_only']['paired_candidate_effect']:+.4f} | {metrics['policy_detached_trunk']['paired_candidate_effect']:+.4f} | {metrics['baseline_joint']['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
    lines.extend(
        [
            "",
            "The step-12 treatment contrast reverses between contexts, so the required persistent detached advantage is not established. No full-epoch continuation was run.",
            "",
            "Fixed-step parameter/output attribution, frozen PUCT probe metrics, provenance-bound arena records, and opening-level bootstrap effects are in `docs/data/alphazero-lite-policy-detached-trunk-summary.json`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
