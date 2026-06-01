#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
    run_corrected_guard_kill_gate,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import load_config
from ml.alphazero_lite.run_guard_safe_opening_full_pipeline_interaction_trace import (
    build_train_spec,
)
from ml.alphazero_lite.run_guard_safe_opening_low_epoch_drift_trace import (
    artifact_row_id,
    choose_device,
    current_checkpoint_path,
    evaluate_dataset_metrics,
    export_checkpoint_artifact,
    train_one_epoch,
)
from ml.alphazero_lite.run_opening_017_puct_target_generation_audit import (
    AUDIT_ROW_IDS,
    DEFAULT_SELF_PLAY_PATH,
    VariantSpec,
    extract_audited_rows,
    merged_reference_rows,
    read_jsonl,
    run_variant_for_row,
    summarize_variant_runs,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
)


DEFAULT_CONFIG_PATH = Path(
    "ml/alphazero_lite/configs/exp_v3_guard_safe_opening_selected_w1_denoised_targets.json"
)
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELECTED_ARTIFACT = Path(
    "/tmp/azlite_guard_safe_opening_replay/"
    "family_leave_one_out_without_opening_extra_turn_overbias.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_denoised_self_play_targets")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "denoised_self_play_targets_summary.json"
DEFAULT_REPORT_PATH = Path("docs/alphazero-lite-denoised-self-play-targets-results.md")
SCHEMA = "azlite_denoised_self_play_targets_v1"
TARGET_EPOCHS = (1, 2, 4)
CRITICAL_ROWS = (
    "capture_available-002",
    "capture_available-006",
    "capture_available-007",
)
TRACE_ROWS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_files: tuple[Path, ...]
    replay_weights: tuple[int, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument("--noisy-self-play", type=Path, default=DEFAULT_SELF_PLAY_PATH)
    parser.add_argument(
        "--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT
    )
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def distribution_text(counter: dict[int, int]) -> str:
    if not counter:
        return "-"
    parts = [f"{move}:{count}" for move, count in sorted(counter.items())]
    return "{" + ", ".join(parts) + "}"


def build_target_specs() -> list[tuple[str, VariantSpec]]:
    return [
        (
            "old_noisy_target_generation",
            VariantSpec(
                name="baseline_self_play_target",
                simulations=192,
                dirichlet_epsilon=0.3,
                policy_target_mode="sharpened",
                search_options_kind="self_play",
            ),
        ),
        (
            "denoised_target_generation",
            VariantSpec(
                name="no_dirichlet",
                simulations=192,
                dirichlet_epsilon=0.0,
                policy_target_mode="sharpened",
                search_options_kind="self_play",
            ),
        ),
        (
            "denoised_target_generation_higher_sims_384",
            VariantSpec(
                name="higher_sims_384",
                simulations=384,
                dirichlet_epsilon=0.0,
                policy_target_mode="default",
                search_options_kind="self_play",
            ),
        ),
    ]


def target_quality_notes(*, row_id: str, mode: str, is_top: bool) -> str:
    notes: list[str] = []
    if mode == "old_noisy_target_generation":
        notes.append("matches_prior_audit_baseline")
    elif row_id == "capture_available-003":
        notes.append(
            "matches_prior_audit_003_still_requires_higher_sims"
            if not is_top
            else "better_than_prior_audit"
        )
    elif row_id == "opening_plies_1_8-017":
        notes.append(
            "matches_prior_audit_017_remains_unstable"
            if not is_top
            else "unexpectedly_stable_vs_prior_audit"
        )
    else:
        notes.append(
            "matches_prior_audit_improvement" if is_top else "worse_than_prior_audit"
        )
    return ", ".join(notes)


def run_target_quality_audit(
    *,
    reference_rows: dict[str, dict[str, Any]],
    noisy_self_play_rows: list[dict[str, Any]],
    current_artifact: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bool, bool]:
    from ml.alphazero_lite.arena import ArtifactEvaluator

    audited_rows, _audited_row_map, _canonical = extract_audited_rows(
        reference_rows=reference_rows,
        self_play_rows=noisy_self_play_rows,
    )
    for row in audited_rows:
        row["state"] = dict(reference_rows[row["row_id"]]["state"])
    evaluator = ArtifactEvaluator(current_artifact)
    table_rows: list[dict[str, Any]] = []
    summaries_by_mode: dict[str, dict[str, Any]] = {
        row_id: {} for row_id in AUDIT_ROW_IDS
    }
    for mode_name, spec in build_target_specs():
        for row in audited_rows:
            runs = run_variant_for_row(evaluator=evaluator, row=row, spec=spec)
            summary = summarize_variant_runs(row=row, variant_name=mode_name, runs=runs)
            summaries_by_mode[row["row_id"]][mode_name] = summary
            is_top = bool(summary["selected_target_matches_corrected_reference"])
            table_rows.append(
                {
                    "row_id": row["row_id"],
                    "mode": mode_name,
                    "simulations": int(summary["simulations"]),
                    "corrected_reference_move": int(row["corrected_reference_move"]),
                    "top_target_move": summary["majority_top_move"],
                    "corrected_reference_mass": float(
                        summary["average_corrected_reference_mass"]
                    ),
                    "target_entropy": float(summary["target_entropy"]),
                    "corrected_reference_is_top": is_top,
                    "notes": target_quality_notes(
                        row_id=row["row_id"], mode=mode_name, is_top=is_top
                    ),
                }
            )
    critical_pass = all(
        bool(
            summaries_by_mode[row_id]["denoised_target_generation"][
                "selected_target_matches_corrected_reference"
            ]
        )
        for row_id in CRITICAL_ROWS
    )
    row_003_requires_more_sims = not bool(
        summaries_by_mode["capture_available-003"]["denoised_target_generation"][
            "selected_target_matches_corrected_reference"
        ]
    )
    row_017_unstable = not bool(
        summaries_by_mode["opening_plies_1_8-017"]["denoised_target_generation"][
            "selected_target_matches_corrected_reference"
        ]
    )
    return (
        table_rows,
        summaries_by_mode,
        critical_pass,
        row_003_requires_more_sims or row_017_unstable,
    )


def run_diagnostic_self_play(*, root: Path, config_path: Path) -> Path:
    config = load_config(config_path)
    run_id = str(config["run_id"])
    versions_dir = Path(str(config["versions_dir"]))
    command = [
        str(root / ".venv/bin/python"),
        "ml/alphazero_lite/pipeline.py",
        "--config",
        str(config_path),
        "--skip-step",
        "perspective_audit",
        "--skip-step",
        "train",
        "--skip-step",
        "export_artifact",
    ]
    subprocess.run(command, cwd=root, check=True)
    return versions_dir / f"{run_id}-iter1" / "self_play.jsonl"


def scan_self_play_dataset(
    *,
    rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    canonical_to_row_id = {
        str(reference_rows[row_id]["canonical_state"]): row_id
        for row_id in AUDIT_ROW_IDS
    }
    aggregates = {
        row_id: {"count": 0, "mass_sum": 0.0, "top_moves": {}}
        for row_id in AUDIT_ROW_IDS
    }
    for row in rows:
        raw_state = decode_state(list(row["state"]))
        row_id = canonical_to_row_id.get(canonical_state_key(raw_state))
        if row_id is None:
            continue
        corrected_move = int(reference_rows[row_id]["reference_move"])
        policy = list(row.get("policy") or [])
        if len(policy) != 6:
            continue
        legal_moves = [
            int(move) for move in KalahGame.from_state(raw_state).possible_moves()
        ]
        top_move = min(legal_moves, key=lambda move: (-float(policy[move]), move))
        aggregates[row_id]["count"] += 1
        aggregates[row_id]["mass_sum"] += float(policy[corrected_move])
        top_moves = aggregates[row_id]["top_moves"]
        top_moves[top_move] = int(top_moves.get(top_move, 0)) + 1
    table_rows: list[dict[str, Any]] = []
    for row_id in AUDIT_ROW_IDS:
        count = int(aggregates[row_id]["count"])
        average_reference_mass = None
        if count > 0:
            average_reference_mass = aggregates[row_id]["mass_sum"] / float(count)
        notes = "ok" if count > 0 else "no_exact_hits"
        table_rows.append(
            {
                "row_id": row_id,
                "dataset": dataset_name,
                "hit_count": count,
                "corrected_reference_move": int(
                    reference_rows[row_id]["reference_move"]
                ),
                "average_reference_mass": average_reference_mass,
                "top_target_distribution": dict(aggregates[row_id]["top_moves"]),
                "notes": notes,
            }
        )
    return table_rows


def current_metadata(current_artifact: Path) -> dict[str, Any]:
    return json.loads((current_artifact / "metadata.json").read_text(encoding="utf-8"))


def build_row_infos(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    row_infos: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row_infos.append({"row_id": artifact_row_id(row)})
    return row_infos


def run_training_trace(
    *,
    spec: TraceSpec,
    init_checkpoint: Path,
    current_artifact: Path,
    train_spec: Any,
    output_root: Path,
    device: torch.device,
    seed: int,
    reference_artifact: Path,
    fallback_reference_artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(seed)
    compact_x, compact_p, compact_v, replay_indexes = load_jsonl_replay(
        list(spec.data_files),
        list(spec.replay_weights),
        policy_target_mode=train_spec.policy_target_mode,
        value_target_mode=train_spec.value_target_mode,
    )
    model = PolicyValueNet(
        hidden_sizes=train_spec.hidden_sizes,
        model_type=train_spec.model_type,
        input_size=input_size_for_encoding(train_spec.input_encoding),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    model = model.to(device)
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)
    checkpoints_root = output_root / "checkpoints" / spec.name
    exports_root = output_root / "exports" / spec.name
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    row_infos = build_row_infos(spec.data_files)
    current_meta = current_metadata(current_artifact)
    for epoch in range(1, max(TARGET_EPOCHS) + 1):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v.astype(np.float32),
            replay_indexes=replay_indexes,
            batch_size=train_spec.batch_size,
            device=device,
            value_loss_weight=train_spec.value_loss_weight,
            value_loss=train_spec.value_loss,
            huber_delta=train_spec.huber_delta,
            grad_clip=train_spec.grad_clip,
        )
        if epoch not in TARGET_EPOCHS:
            continue
        checkpoint_path = checkpoints_root / f"epoch_{epoch}.npz"
        np.savez(checkpoint_path, **checkpoint_from_model(model))
        export_dir = export_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            export_dir=exports_root / f"epoch_{epoch}",
            current_metadata=current_meta,
            version=f"{spec.name}-epoch-{epoch}",
        )
        guard_payload = run_corrected_guard_kill_gate(
            candidate_path=export_dir,
            reference_artifact=reference_artifact,
            fallback_reference_artifact=fallback_reference_artifact,
            seed=seed + epoch,
        )
        gate_pass = bool(guard_payload["pass"])
        for row in guard_payload["rows"]:
            guard_rows.append(
                {
                    "trace_name": spec.name,
                    "epoch": epoch,
                    "row_id": row["row_id"],
                    "corrected_reference_move": row["corrected_reference_move"],
                    "selected_move_384": row["selected_move_384"],
                    "selected_move_1200": row["selected_move_1200"],
                    "reference_visit_share_384": row.get("reference_visit_share_384"),
                    "reference_visit_share_1200": row.get("reference_visit_share_1200"),
                    "selected_minus_reference_q_margin_384": row.get(
                        "selected_minus_reference_q_margin_384"
                    ),
                    "selected_minus_reference_q_margin_1200": row.get(
                        "selected_minus_reference_q_margin_1200"
                    ),
                    "row_pass": bool(row["pass"]),
                    "gate_pass": gate_pass,
                    "notes": row.get("notes", "ok"),
                }
            )
        metrics = evaluate_dataset_metrics(
            model=model,
            compact_x=compact_x,
            compact_p=compact_p,
            compact_v=compact_v.astype(np.float32),
            row_infos=row_infos,
            device=device,
            value_loss_weight=train_spec.value_loss_weight,
            value_loss=train_spec.value_loss,
            huber_delta=train_spec.huber_delta,
            initial_state=initial_state,
        )
        metrics_rows.append({"trace_name": spec.name, "epoch": epoch, **metrics})
    return metrics_rows, guard_rows


def decide_next_action(
    *,
    target_quality_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    denoised = {
        row["row_id"]: row
        for row in target_quality_rows
        if row["mode"] == "denoised_target_generation"
    }
    descendants_safe = all(
        bool(denoised[row_id]["corrected_reference_is_top"]) for row_id in TRACE_ROWS
    )
    row_003_fails = not bool(
        denoised["capture_available-003"]["corrected_reference_is_top"]
    )
    row_017_unstable = not bool(
        denoised["opening_plies_1_8-017"]["corrected_reference_is_top"]
    )
    critical_fail = any(
        not bool(denoised[row_id]["corrected_reference_is_top"])
        for row_id in CRITICAL_ROWS
    )
    any_gate_pass = any(bool(row["gate_pass"]) for row in guard_rows)
    if critical_fail:
        return (
            "target_noise_not_the_only_cause",
            "full trajectory target writeout audit, especially tree reuse and target extraction.",
        )
    if row_003_fails:
        return (
            "denoised_targets_partial_003_needs_more_sims",
            "test denoised targets plus opening minimum 384 sims, diagnostic only.",
        )
    if row_017_unstable and descendants_safe:
        return (
            "predecessor_unstable_descendants_safe",
            "exclude 017 from hard target gates and track it as unstable, while continuing with descendant guard validation.",
        )
    if any_gate_pass:
        return (
            "denoised_targets_ready_for_controlled_lane",
            "run one controlled production-scale lane with denoised targets and corrected guard kill gate before arena.",
        )
    return (
        "training_objective_or_mix_still_regresses",
        "inspect loss weighting / sampling mix before production training.",
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Denoised Self-Play Targets Results",
        "",
        "## 1. Context",
        "",
        "- Commit `2dc740b` identified noisy self-play policy targets as the leading cause of corrected guard regression.",
        "- This run added optional denoised target generation while leaving default production behavior unchanged.",
        "- This run stayed diagnostic-only: no arena, no promotion, no overwrite of `storage/ai/alphazero_lite/current`, and no broad replay-weight sweep.",
        "",
        "## 2. Why Noisy Targets Are Now The Suspected Cause",
        "",
        "- The prior opening-017 audit showed root Dirichlet noise restored bad target mass on corrected descendants.",
        "- The corrective patch trace still regressed under training, which pointed back to the self-play labels themselves rather than replay anchoring.",
        "",
        "## 3. Self-Play Implementation Change",
        "",
        "- Added `--policy-target-noise-mode noisy|denoised` with default `noisy`.",
        "- In `denoised` mode, action sampling can still use root Dirichlet noise, but the stored policy target is generated from a separate no-noise root search.",
        "- Added optional `--write-root-target-telemetry` row metadata for root visits, priors, and stored targets.",
        "",
        "## 4. Target-Quality Audit",
        "",
        "| row_id | mode | simulations | corrected_reference_move | top_target_move | corrected_reference_mass | target_entropy | corrected_reference_is_top | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["target_quality_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['mode']} | {row['simulations']} | {row['corrected_reference_move']} | {row['top_target_move'] if row['top_target_move'] is not None else '-'} | {format_float(row['corrected_reference_mass'])} | {format_float(row['target_entropy'])} | {str(bool(row['corrected_reference_is_top'])).lower()} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Diagnostic Self-Play Scan",
            "",
            "| row_id | dataset | hit_count | corrected_reference_move | average_reference_mass | top_target_distribution | notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["self_play_scan_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['dataset']} | {row['hit_count']} | {row['corrected_reference_move']} | {format_float(row['average_reference_mass'])} | `{distribution_text(row['top_target_distribution'])}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Low-Cost Training Trace",
            "",
            f"- `noisy_self_play_reproduction`: data `{summary['trace_inputs']['noisy_self_play_reproduction']}` epochs `{summary['target_epochs']}`.",
            f"- `denoised_self_play_only`: data `{summary['trace_inputs']['denoised_self_play_only']}` epochs `{summary['target_epochs']}`.",
            f"- `denoised_self_play_plus_selected_artifact_w1`: data `{summary['trace_inputs']['denoised_self_play_plus_selected_artifact_w1']}` epochs `{summary['target_epochs']}`.",
            "",
            "## 7. Corrected Guard Kill-Gate Results",
            "",
            "| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["guard_rows"]:
        lines.append(
            f"| {row['trace_name']} | {row['epoch']} | {row['row_id']} | {row['corrected_reference_move']} | {row['selected_move_384']} | {row['selected_move_1200']} | {format_float(row['reference_visit_share_384'])} | {format_float(row['reference_visit_share_1200'])} | {str(bool(row['row_pass'])).lower()} | {str(bool(row['gate_pass'])).lower()} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Interpretation",
            "",
            f"- Classification: `{summary['classification']}`.",
            f"- `capture_available-003` still needs higher sims: `{str(bool(summary['row_003_still_needs_more_sims'])).lower()}`.",
            f"- `opening_plies_1_8-017` remains unstable: `{str(bool(summary['row_017_remains_unstable'])).lower()}`.",
            f"- Low-cost trace achieved a full corrected guard pass: `{str(bool(summary['any_gate_pass'])).lower()}`.",
            "",
            "## 9. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_path(root, args.config)
    current_artifact = resolve_path(root, args.current_artifact)
    noisy_self_play_path = resolve_path(root, args.noisy_self_play)
    selected_artifact = resolve_path(root, args.selected_artifact)
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    output_root = resolve_path(root, args.output_root)
    summary_path = resolve_path(root, args.summary_path)
    report_path = resolve_path(root, args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    reference_rows = merged_reference_rows(
        reference_artifact, fallback_reference_artifact
    )
    noisy_self_play_rows = read_jsonl(noisy_self_play_path)
    target_quality_rows, target_summaries, critical_pass, unstable_or_003 = (
        run_target_quality_audit(
            reference_rows=reference_rows,
            noisy_self_play_rows=noisy_self_play_rows,
            current_artifact=current_artifact,
        )
    )

    denoised_self_play_path = run_diagnostic_self_play(
        root=root, config_path=config_path
    )
    denoised_self_play_rows = read_jsonl(denoised_self_play_path)
    self_play_scan_rows = scan_self_play_dataset(
        rows=noisy_self_play_rows,
        reference_rows=reference_rows,
        dataset_name="pr36_noisy_self_play",
    ) + scan_self_play_dataset(
        rows=denoised_self_play_rows,
        reference_rows=reference_rows,
        dataset_name="denoised_diagnostic_self_play",
    )

    train_spec = build_train_spec(config)
    init_checkpoint = current_checkpoint_path(current_artifact, output_root)
    device = choose_device(args.device)
    guard_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trace_inputs = {
        "noisy_self_play_reproduction": [str(noisy_self_play_path)],
        "denoised_self_play_only": [str(denoised_self_play_path)],
        "denoised_self_play_plus_selected_artifact_w1": [
            str(denoised_self_play_path),
            str(selected_artifact),
        ],
    }
    if critical_pass:
        traces = [
            TraceSpec(
                name="noisy_self_play_reproduction",
                data_files=(noisy_self_play_path,),
                replay_weights=(1,),
            ),
            TraceSpec(
                name="denoised_self_play_only",
                data_files=(denoised_self_play_path,),
                replay_weights=(1,),
            ),
            TraceSpec(
                name="denoised_self_play_plus_selected_artifact_w1",
                data_files=(denoised_self_play_path, selected_artifact),
                replay_weights=(1, 1),
            ),
        ]
        for index, trace in enumerate(traces):
            trace_metrics, trace_guard_rows = run_training_trace(
                spec=trace,
                init_checkpoint=init_checkpoint,
                current_artifact=current_artifact,
                train_spec=train_spec,
                output_root=output_root,
                device=device,
                seed=42 + (index * 100),
                reference_artifact=reference_artifact,
                fallback_reference_artifact=fallback_reference_artifact,
            )
            metric_rows.extend(trace_metrics)
            guard_rows.extend(trace_guard_rows)

    classification, recommended_next_action = decide_next_action(
        target_quality_rows=target_quality_rows,
        guard_rows=guard_rows,
    )
    summary = {
        "schema": SCHEMA,
        "config": config,
        "target_quality_rows": target_quality_rows,
        "self_play_scan_rows": self_play_scan_rows,
        "metric_rows": metric_rows,
        "guard_rows": guard_rows,
        "classification": classification,
        "recommended_next_action": recommended_next_action,
        "row_003_still_needs_more_sims": not bool(
            target_summaries["capture_available-003"]["denoised_target_generation"][
                "selected_target_matches_corrected_reference"
            ]
        ),
        "row_017_remains_unstable": not bool(
            target_summaries["opening_plies_1_8-017"]["denoised_target_generation"][
                "selected_target_matches_corrected_reference"
            ]
        ),
        "critical_audit_pass": critical_pass,
        "any_gate_pass": any(bool(row["gate_pass"]) for row in guard_rows),
        "target_epochs": list(TARGET_EPOCHS),
        "trace_inputs": trace_inputs,
        "paths": {
            "current_artifact": str(current_artifact),
            "noisy_self_play": str(noisy_self_play_path),
            "denoised_self_play": str(denoised_self_play_path),
            "selected_artifact": str(selected_artifact),
        },
        "constraints": {
            "arena_run": False,
            "promotion_run": False,
            "overwrite_current": False,
            "broad_replay_weight_sweep": False,
        },
    }
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": classification,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
