#!/usr/bin/env python3
"""Run original-data continuation epoch sweep from iter0_reference.

Experiment lanes:
  1. iter0_reference_eval_only (no training)
  2. continue_lr_3e_5_epochs_1_to_8 (LR=3e-5)
  3. continue_lr_1e_5_epochs_1_to_8 (LR=1e-5)

Does not promote, does not overwrite current, does not add curriculum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")

TARGET_EPOCHS = [1, 2, 3, 4, 6, 8]
BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,256:768"


def _python() -> str:
    if Path(VENV_PYTHON).is_file():
        return VENV_PYTHON
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_checkpoint(npz_path: Path) -> str:
    return sha256_file(npz_path)


def compute_param_delta_norm(
    npz_path: Path, reference_npz: Path
) -> tuple[float, float]:
    """Compute L2 norm of parameter delta vs reference checkpoint."""
    import numpy as np

    ckpt = np.load(npz_path)
    ref = np.load(reference_npz)

    total_sq = 0.0
    ref_sq = 0.0
    for key in sorted(ref.files):
        if key in ckpt:
            delta = ckpt[key] - ref[key]
            total_sq += float(np.sum(delta**2))
            ref_sq += float(np.sum(ref[key] ** 2))

    delta_norm = float(np.sqrt(total_sq))
    ref_norm = float(np.sqrt(ref_sq))
    rel_delta = (delta_norm / ref_norm * 100.0) if ref_norm > 0 else 0.0
    return delta_norm, rel_delta


def run_train(
    *,
    data_files: str,
    replay_weights: str,
    init_checkpoint: str,
    out: str,
    top_k_dir: str,
    lr: float,
    epochs: int,
    save_epochs: str,
    lr_scheduler: str = "none",
) -> dict:
    """Run train.py and return parsed metrics."""
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--data-files",
        data_files,
        "--replay-weights",
        replay_weights,
        "--init-checkpoint",
        init_checkpoint,
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--hidden-sizes",
        "96,3",
        "--epochs",
        str(epochs),
        "--batch-size",
        "512",
        "--lr",
        str(lr),
        "--value-loss",
        "huber",
        "--value-loss-weight",
        "0.3",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "0",
        "--top-k-dir",
        top_k_dir,
        "--out",
        out,
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--lr-scheduler",
        lr_scheduler,
        "--seed",
        "42",
        "--save-epochs",
        save_epochs,
    ]

    print(f"[train] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - t0
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        print(f"[train] FAILED after {elapsed:.0f}s", flush=True)
        print(f"[train] stdout: {stdout[-2000:]}", flush=True)
        print(f"[train] stderr: {stderr[-2000:]}", flush=True)
        raise RuntimeError(f"train.py failed with return code {result.returncode}")

    metrics: dict[str, object] = {"training_elapsed_s": elapsed}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("policy_loss="):
            metrics["policy_loss"] = float(line.split("=")[1])
        elif line.startswith("value_loss="):
            metrics["value_loss"] = float(line.split("=")[1])
        elif line.startswith("best_val_loss="):
            metrics["best_val_loss"] = float(line.split("=")[1])
        elif line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            epoch_part = parts[0].split("=")[1]
            path_part = parts[1].split("=")[1]
            metrics[f"epoch_{epoch_part}_path"] = path_part
    print(
        f"[train] done {elapsed:.0f}s policy_loss={metrics.get('policy_loss', '?'):.6f}",
        flush=True,
    )
    return metrics


def export_checkpoint(
    checkpoint_path: str,
    out_dir: str,
    version: str,
    policy_loss: float = 0.0,
    value_loss: float = 0.0,
    model_type: str = "residual_v3",
    input_encoding: str = "kalah_v3",
) -> None:
    """Export a checkpoint to artifact format."""
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        checkpoint_path,
        "--out-dir",
        out_dir,
        "--version",
        version,
        "--model-type",
        model_type,
        "--input-encoding",
        input_encoding,
        "--rules-version",
        input_encoding,
        "--policy-loss",
        str(policy_loss),
        "--value-loss",
        str(value_loss),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export_artifact failed: {result.stderr}")
    print(f"[export] {out_dir}", flush=True)


def run_opening_suite_benchmark(
    *,
    workdir: str,
    suite: str,
    current: str,
    candidates: str,
    budget_pairs: str,
    games_per_opening: int = 2,
    seed: int = 42,
    root_policy_mode: str = "deterministic",
    workers: int = 4,
    timeout: int = 7200,
) -> dict:
    """Run opening-suite seat-aware benchmark."""
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        workdir,
        "--suite",
        suite,
        "--current",
        current,
        "--candidates",
        candidates,
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--root-policy-mode",
        root_policy_mode,
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout + 300,
    )
    if result.returncode != 0:
        print(f"[eval] FAILED: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"benchmark failed: {result.stderr}")

    report_path = Path(workdir) / "temperature_benchmark_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {}


def run_default_gate(
    *,
    candidate_path: str,
    current_path: str,
    out: str,
    games: int = 60,
    seed: int = 42,
    workers: int = 4,
    budget_pairs: str = "384:256,768:256,768:768,1200:1200,256:768",
) -> dict:
    """Run default deterministic opening gate."""
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        candidate_path,
        "--current-path",
        current_path,
        "--out",
        out,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        budget_pairs,
    ]
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        print(f"[gate] FAILED: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"gate failed: {result.stderr}")

    out_path = Path(out)
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return {}


def extract_ds_summary(report: dict, candidate: str) -> dict[str, float | None]:
    """Extract DS values from benchmark report for a candidate."""
    result: dict[str, float | None] = {
        "384:256": None,
        "768:256": None,
        "768:768": None,
        "1200:1200": None,
        "256:768": None,
    }

    temp_reports = report.get("temperature_reports", [])
    for tr in temp_reports:
        for sr in tr.get("seed_reports", []):
            for cr in sr.get("candidate_reports", []):
                if cr.get("candidate") == candidate or candidate in cr.get(
                    "candidate_path", ""
                ):
                    budget_results = cr.get("budget_results", {})
                    label_to_budget = {
                        "standard": "384:256",
                        "challenger_768_vs_256": "768:256",
                        "equal_768": "768:768",
                        "equal_high": "1200:1200",
                        "current_high_asymmetry": "256:768",
                    }
                    for blabel, bkey in label_to_budget.items():
                        br = budget_results.get(blabel, {})
                        if br:
                            result[bkey] = br.get("ds")
                    return result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_control_continuation_sweep")
    parser.add_argument(
        "--data-files",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,"
        "/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument("--replay-weights", default="4,1")
    parser.add_argument(
        "--init-checkpoint",
        default="/tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz",
    )
    parser.add_argument(
        "--init-artifact",
        default="/tmp/azlite_iterative_random_replay/iter0_candidate_artifact",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--suite-medium",
        default="/tmp/azlite_opening_suite/medium_eval.jsonl",
    )
    parser.add_argument(
        "--suite-large",
        default="/tmp/azlite_opening_suite/large_eval.jsonl",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval-medium", action="store_true")
    parser.add_argument("--skip-eval-large", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = Path(args.init_checkpoint)
    init_artifact = Path(args.init_artifact)

    # Training lanes
    training_configs = [
        {
            "lane": "lr3e5",
            "lr": 3e-5,
            "lr_scheduler": "none",
            "dir": workdir / "lr3e5",
        },
        {
            "lane": "lr1e5",
            "lr": 1e-5,
            "lr_scheduler": "none",
            "dir": workdir / "lr1e5",
        },
    ]

    # ============================================================
    # Phase 1: Training
    # ============================================================
    if not args.skip_training:
        save_epochs_str = ",".join(str(e) for e in TARGET_EPOCHS)
        for tc in training_configs:
            tc["dir"].mkdir(parents=True, exist_ok=True)
            out_path = str(tc["dir"] / "checkpoint.npz")
            print(f"\n{'=' * 60}")
            print(f"Training lane: {tc['lane']} LR={tc['lr']} epochs={args.epochs}")
            print(f"{'=' * 60}", flush=True)

            train_metrics = run_train(
                data_files=args.data_files,
                replay_weights=args.replay_weights,
                init_checkpoint=str(init_checkpoint),
                out=out_path,
                top_k_dir=str(tc["dir"]),
                lr=tc["lr"],
                epochs=args.epochs,
                save_epochs=save_epochs_str,
                lr_scheduler=tc["lr_scheduler"],
            )
            tc["train_metrics"] = train_metrics

    # ============================================================
    # Phase 2: Export checkpoints to artifacts
    # ============================================================
    all_candidates: dict[str, str] = {}
    all_candidates["iter0_reference"] = str(init_artifact)

    for tc in training_configs:
        target_epochs_list = TARGET_EPOCHS
        for ep in target_epochs_list:
            lane_name = f"{tc['lane']}_ep{ep}"
            checkpoint_path = tc["dir"] / f"checkpoint_epoch{ep}.npz"
            artifact_dir = tc["dir"] / f"artifact_epoch{ep}"

            if not checkpoint_path.exists():
                print(f"[warn] checkpoint not found: {checkpoint_path}", flush=True)
                continue

            if (
                not artifact_dir.exists()
                or not (artifact_dir / "weights.json").exists()
            ):
                train_metrics = tc.get("train_metrics", {})
                policy_loss = train_metrics.get("policy_loss", 0.0)
                value_loss = train_metrics.get("value_loss", 0.0)
                export_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    out_dir=str(artifact_dir),
                    version=f"{tc['lane']}_ep{ep}",
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                )

            all_candidates[lane_name] = str(artifact_dir)

    print(f"\nTotal candidates: {len(all_candidates)}")
    for name, path in all_candidates.items():
        print(f"  {name}: {path}")

    # ============================================================
    # Phase 3: Compute parameter deltas
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Parameter delta norms vs iter0_reference")
    print(f"{'=' * 60}", flush=True)

    param_deltas: dict[str, tuple[float, float]] = {}
    for name, artifact_path in all_candidates.items():
        artifact_dir = Path(artifact_path)
        npz_path = artifact_dir / "model.npz"
        if npz_path.exists():
            delta_norm, rel_delta = compute_param_delta_norm(npz_path, init_checkpoint)
            param_deltas[name] = (delta_norm, rel_delta)
            print(f"  {name}: delta_norm={delta_norm:.6f} rel_delta={rel_delta:.2f}%")
        else:
            print(f"  {name}: no model.npz found")

    # ============================================================
    # Phase 4: Medium eval
    # ============================================================
    if not args.skip_eval_medium:
        print(f"\n{'=' * 60}")
        print("Medium Evaluation")
        print(f"{'=' * 60}", flush=True)

        candidates_str = ",".join(all_candidates.values())
        medium_workdir = str(workdir / "eval_medium")

        medium_report = run_opening_suite_benchmark(
            workdir=medium_workdir,
            suite=args.suite_medium,
            current=args.current,
            candidates=candidates_str,
            budget_pairs=BUDGET_PAIRS,
            games_per_opening=args.games_per_opening,
            seed=42,
            root_policy_mode="deterministic",
            workers=args.workers,
            timeout=args.timeout,
        )

        report_path = Path(medium_workdir) / "temperature_benchmark_report.json"
        report_path.write_text(json.dumps(medium_report, indent=2), encoding="utf-8")

    # ============================================================
    # Phase 5: Large eval on best 2-3
    # ============================================================
    if not args.skip_eval_large:
        # Identify best candidates from medium eval
        medium_report_path = (
            Path(workdir) / "eval_medium" / "temperature_benchmark_report.json"
        )
        if medium_report_path.exists():
            medium_report = json.loads(medium_report_path.read_text(encoding="utf-8"))

            candidate_ds: dict[str, float] = {}
            for tr in medium_report.get("temperature_reports", []):
                for sr in tr.get("seed_reports", []):
                    for cr in sr.get("candidate_reports", []):
                        name = cr.get("candidate", "")
                        br = cr.get("budget_results", {})
                        std = br.get("standard", {})
                        ds = std.get("ds")
                        if ds is not None and name:
                            candidate_ds[name] = float(ds)

            sorted_candidates = sorted(
                candidate_ds.items(), key=lambda x: x[1], reverse=True
            )
            print("\nMedium eval ranking by standard DS:")
            for name, ds in sorted_candidates:
                print(f"  {name}: DS={ds:+.4f}")

            best_names = [name for name, _ in sorted_candidates[:3]]
            best_paths = [
                all_candidates[name] for name in best_names if name in all_candidates
            ]

            if best_paths:
                print(f"\nLarge eval on best candidates: {best_names}", flush=True)
                large_workdir = str(workdir / "eval_large")
                large_candidates = ",".join([str(init_artifact)] + best_paths)

                large_report = run_opening_suite_benchmark(
                    workdir=large_workdir,
                    suite=args.suite_large,
                    current=args.current,
                    candidates=large_candidates,
                    budget_pairs=BUDGET_PAIRS,
                    games_per_opening=args.games_per_opening,
                    seed=42,
                    root_policy_mode="deterministic",
                    workers=args.workers,
                    timeout=args.timeout,
                )

                report_path = Path(large_workdir) / "temperature_benchmark_report.json"
                report_path.write_text(
                    json.dumps(large_report, indent=2), encoding="utf-8"
                )

    # ============================================================
    # Phase 6: Default opening gate
    # ============================================================
    if not args.skip_gate:
        print(f"\n{'=' * 60}")
        print("Default Opening Gate")
        print(f"{'=' * 60}", flush=True)

        gate_dir = workdir / "eval_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)

        for name, artifact_path in all_candidates.items():
            if name == "iter0_reference":
                continue

            gate_out = str(gate_dir / f"{name}_default_gate.json")

            if Path(gate_out).exists():
                print(f"[gate] {name}: already exists, skipping", flush=True)
                continue

            try:
                gate_report = run_default_gate(
                    candidate_path=artifact_path,
                    current_path=args.current,
                    out=gate_out,
                    games=60,
                    seed=42,
                    workers=args.workers,
                    budget_pairs=BUDGET_PAIRS,
                )
                classification = gate_report.get("classification", "unknown")
                print(f"[gate] {name}: classification={classification}", flush=True)
            except RuntimeError as e:
                print(f"[gate] {name}: FAILED - {e}", flush=True)

    # ============================================================
    # Phase 7: Compile final report
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Final Summary")
    print(f"{'=' * 60}", flush=True)

    # Build consolidated table
    rows: list[dict[str, object]] = []
    for name, artifact_path in all_candidates.items():
        row: dict[str, object] = {
            "candidate": name,
            "artifact_path": artifact_path,
        }

        # SHA256
        artifact_dir = Path(artifact_path)
        weights_path = artifact_dir / "weights.json"
        if weights_path.exists():
            row["sha256_weights"] = sha256_file(weights_path)

        npz_path = artifact_dir / "model.npz"
        if npz_path.exists():
            row["sha256_model"] = sha256_file(npz_path)

        # Parameter delta
        if name in param_deltas:
            delta_norm, rel_delta = param_deltas[name]
            row["delta_norm"] = delta_norm
            row["rel_delta_pct"] = rel_delta

        # Extract LR and epoch from name
        if "lr3e5" in name:
            row["lr"] = 3e-5
        elif "lr1e5" in name:
            row["lr"] = 1e-5
        else:
            row["lr"] = None

        if "_ep" in name:
            parts = name.split("_ep")
            if len(parts) > 1:
                row["epoch"] = int(parts[-1])
        else:
            row["epoch"] = None

        rows.append(row)

    final_report = {
        "schema": "azlite_control_continuation_sweep_v1",
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "init_artifact_sha256": sha256_file(init_artifact / "weights.json"),
        "current_sha256": sha256_file(Path(args.current) / "weights.json"),
        "dataset_files": args.data_files.split(","),
        "replay_weights": args.replay_weights,
        "target_epochs": TARGET_EPOCHS,
        "candidates": rows,
    }

    report_out = workdir / "continuation_sweep_report.json"
    report_out.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
