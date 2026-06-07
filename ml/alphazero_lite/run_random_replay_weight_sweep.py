#!/usr/bin/env python3
"""Sweep mined-replay weights to determine whether effective sample share is the
limiting factor for random current-vs-classic-MCTS relabeled replay."""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def effective_sample_share(weights: list[int], row_counts: list[int]) -> list[float]:
    expanded = [w * r for w, r in zip(weights, row_counts)]
    total = sum(expanded)
    return [e / total for e in expanded]


def wilson_ci(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    margin = (
        z * math.sqrt(score * (1.0 - score) / n + z * z / (4.0 * n * n)) / denominator
    )
    return (center - margin, center + margin)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], *, cwd: Path, timeout: int = 3600) -> dict:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    duration = round(time.time() - started, 2)
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_s": duration,
        "stdout": completed.stdout[-4000:]
        if len(completed.stdout) > 4000
        else completed.stdout,
        "stderr": completed.stderr[-4000:]
        if len(completed.stderr) > 4000
        else completed.stderr,
    }


def arena_score_from_report(report: dict) -> float:
    games_played = int(report.get("games_played", 0))
    if games_played <= 0:
        return 0.0
    wins = int(report.get("wins", 0))
    draws = int(report.get("draws", 0))
    return (wins + 0.5 * draws) / games_played


def parse_replay_weight_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            raise SystemExit(
                f"invalid replay weight pair: {chunk!r}, expected format like 4:1"
            )
        try:
            a = int(parts[0].strip())
            b = int(parts[1].strip())
        except ValueError:
            raise SystemExit(
                f"invalid replay weight pair: {chunk!r}, values must be integers"
            ) from None
        if a <= 0 or b <= 0:
            raise SystemExit(
                f"invalid replay weight pair: {chunk!r}, weights must be positive"
            )
        pairs.append((a, b))
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep mined-replay weights for random relabeled replay."
    )
    parser.add_argument("--workdir", required=True, help="Root output directory")
    parser.add_argument(
        "--generic-replay", required=True, help="Path to generic bootstrap JSONL"
    )
    parser.add_argument(
        "--random-replay", required=True, help="Path to random mined train JSONL"
    )
    parser.add_argument(
        "--init-current",
        default="storage/ai/alphazero_lite/current",
        help="Path to current weights.json directory",
    )
    parser.add_argument(
        "--replay-weight-pairs",
        default="8:1,4:1,2:1,1:1",
        help="Comma-separated generic:mined weight pairs",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--arena-games", type=int, default=120)
    parser.add_argument("--extended-arena-games", type=int, default=240)
    parser.add_argument("--challenger-simulations", type=int, default=384)
    parser.add_argument("--current-simulations", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args(argv)


def materialize_init_checkpoint(current_dir: Path, out_path: Path) -> Path:
    weights_path = current_dir / "weights.json"
    if weights_path.exists():
        return materialize_weights_json_checkpoint(
            weights_path=weights_path,
            out_path=out_path,
        )
    npz_path = current_dir / "checkpoint.npz"
    if npz_path.exists():
        return npz_path
    model_path = current_dir / "model.npz"
    if model_path.exists():
        return model_path
    raise SystemExit(f"no usable checkpoint found in {current_dir}")


def train_lane(
    *,
    python: str,
    workdir: Path,
    lane_name: str,
    data_files: list[Path],
    replay_weights: list[int],
    init_checkpoint: Path,
    epochs: int,
    batch_size: int,
    save_top_k: int,
    seed: int,
    cwd: Path,
    timeout: int,
) -> dict:
    lane_dir = workdir / lane_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = lane_dir / "checkpoint.npz"

    data_files_arg = ",".join(str(p) for p in data_files)
    weights_arg = ",".join(str(w) for w in replay_weights)

    command = [
        python,
        "ml/alphazero_lite/train.py",
        "--data-files",
        data_files_arg,
        "--replay-weights",
        weights_arg,
        "--init-checkpoint",
        str(init_checkpoint),
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--hidden-sizes",
        "96,3",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        "1e-3",
        "--value-loss",
        "huber",
        "--value-loss-weight",
        "0.3",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        str(save_top_k),
        "--top-k-dir",
        str(lane_dir),
        "--out",
        str(checkpoint_out),
        "--device",
        "auto",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--seed",
        str(seed),
    ]

    result = run_command(command, cwd=cwd, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result

    result["status"] = "completed"

    metrics = {}
    for line in result.get("stdout", "").splitlines():
        for key in ("policy_loss", "value_loss", "best_val_loss"):
            if line.startswith(f"{key}="):
                try:
                    metrics[key] = float(line.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass

    topk_snapshots = []
    for idx in range(1, save_top_k + 1):
        candidate = lane_dir / f"checkpoint.top{idx}.npz"
        if candidate.exists():
            topk_snapshots.append(
                {
                    "rank": idx,
                    "path": str(candidate),
                    "sha256": sha256_hex(candidate),
                }
            )

    epoch_metrics: dict[str, float | str | None] = {
        "policy_loss": metrics.get("policy_loss"),
        "value_loss": metrics.get("value_loss"),
        "best_val_loss": metrics.get("best_val_loss"),
    }

    return {
        **result,
        "metrics": epoch_metrics,
        "topk_checkpoints": topk_snapshots,
        "checkpoint_path": str(checkpoint_out),
    }


def export_artifact(
    *,
    python: str,
    checkpoint_path: Path,
    artifact_dir: Path,
    version: str,
    cwd: Path,
    timeout: int,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "ml/alphazero_lite/export_artifact.py",
        "--checkpoint",
        str(checkpoint_path),
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
    ]
    result = run_command(command, cwd=cwd, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    result["status"] = "completed"
    result["artifact_dir"] = str(artifact_dir)
    result["version"] = version
    return result


def run_arena(
    *,
    python: str,
    challenger_path: Path,
    current_path: str,
    games: int,
    challenger_simulations: int,
    current_simulations: int,
    report_path: Path,
    workers: int,
    seed: int,
    cwd: Path,
    timeout: int,
) -> dict:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(challenger_path),
        "--current",
        str(current_path),
        "--games",
        str(games),
        "--challenger-simulations",
        str(challenger_simulations),
        "--current-simulations",
        str(current_simulations),
        "--out",
        str(report_path),
        "--min-score",
        "0.0",
        "--workers",
        str(workers),
        "--seed",
        str(seed),
    ]
    result = run_command(command, cwd=cwd, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    result["status"] = "completed"
    if report_path.exists():
        report = load_json(report_path)
        score = arena_score_from_report(report)
        n = int(report.get("games_played", games))
        ci_low, ci_high = wilson_ci(score, n)
        notes = report.get("notes", {}) if isinstance(report.get("notes"), dict) else {}
        result["arena"] = {
            "score": round(score, 4),
            "games_played": n,
            "wins": report.get("wins"),
            "losses": report.get("losses"),
            "draws": report.get("draws"),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "move_time_mean_ms": notes.get("move_time_mean_ms"),
            "move_time_p95_ms": notes.get("move_time_p95_ms"),
        }
    else:
        result["arena"] = {"error": "report not found"}
    return result


def run_gate(
    *,
    python: str,
    candidate_path: Path,
    out_path: Path,
    current_path: str,
    arena_games: int,
    cwd: Path,
    timeout: int,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "script/ai/local_promotion_gate",
        "--candidate-path",
        str(candidate_path),
        "--current-path",
        current_path,
        "--arena-games",
        str(arena_games),
        "--out",
        str(out_path),
    ]
    result = run_command(command, cwd=cwd, timeout=timeout)
    if result["returncode"] != 0:
        result["status"] = "failed"
        if out_path.exists():
            gate_report = load_json(out_path)
            result["gate"] = {
                "passed": gate_report.get("passed", False),
                "arena_score": gate_report.get("arena_score"),
                "failure_reasons": gate_report.get("failure_reasons", []),
            }
        return result
    result["status"] = "completed"
    if out_path.exists():
        gate_report = load_json(out_path)
        result["gate"] = {
            "passed": gate_report.get("passed", False),
            "arena_score": gate_report.get("arena_score"),
            "hard_score": gate_report.get("hard_score"),
            "candidate_mcts_score": gate_report.get("candidate_mcts_score"),
            "current_mcts_score": gate_report.get("current_mcts_score"),
            "failure_reasons": gate_report.get("failure_reasons", []),
        }
    return result


def determine_effective_sample_shares(
    generic_rows: int,
    mined_rows: int,
    lane_weights: list[tuple[int, int]],
) -> dict[str, float]:
    shares = {}
    for gw, mw in lane_weights:
        label = f"weights_{gw}_{mw}"
        expanded = [gw * generic_rows, mw * mined_rows]
        total = sum(expanded)
        shares[f"{label}_generic"] = expanded[0] / total
        shares[f"{label}_mined"] = expanded[1] / total
    return shares


def build_markdown_report(
    *,
    dataset_info: dict,
    baseline: dict,
    lanes: dict,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# AlphaZero-Lite Random Replay Weight Sweep Results",
        "",
        "**Date:** 2026-06-07",
        "",
        "## Classification",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    python = python_bin(root)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    generic_replay = Path(args.generic_replay)
    random_replay = Path(args.random_replay)
    init_current = Path(args.init_current)

    if not generic_replay.exists():
        raise SystemExit(f"missing generic replay: {generic_replay}")
    if not random_replay.exists():
        raise SystemExit(f"missing random replay: {random_replay}")
    if not init_current.exists():
        raise SystemExit(f"missing init current: {init_current}")

    generic_sha256 = sha256_hex(generic_replay)
    random_sha256 = sha256_hex(random_replay)
    generic_rows = count_jsonl_rows(generic_replay)
    random_rows = count_jsonl_rows(random_replay)

    print(
        f"generic replay: {generic_replay} sha256={generic_sha256} rows={generic_rows}"
    )
    print(
        f"random mined replay: {random_replay} sha256={random_sha256} rows={random_rows}"
    )

    weight_pairs = parse_replay_weight_pairs(args.replay_weight_pairs)

    init_checkpoint = materialize_init_checkpoint(
        init_current, workdir / "init_checkpoint.npz"
    )
    init_sha256 = sha256_hex(init_checkpoint)
    print(f"init checkpoint: {init_checkpoint} sha256={init_sha256}")

    dataset_info = {
        "generic_replay": {
            "path": str(generic_replay),
            "sha256": generic_sha256,
            "rows": generic_rows,
        },
        "random_mined_replay": {
            "path": str(random_replay),
            "sha256": random_sha256,
            "rows": random_rows,
        },
        "init_checkpoint": {
            "path": str(init_checkpoint),
            "sha256": init_sha256,
            "source": str(init_current),
        },
    }

    lane_definitions: list[dict] = [
        {
            "name": "baseline_replay",
            "data_files": [generic_replay],
            "replay_weights": [4],
            "label": "baseline (generic-only weight=4)",
        }
    ]
    for gw, mw in weight_pairs:
        lane_definitions.append(
            {
                "name": f"random_weight_{gw}_{mw}",
                "data_files": [generic_replay, random_replay],
                "replay_weights": [gw, mw],
                "label": f"random-mined weights={gw}:{mw}",
            }
        )

    effective_shares = {}
    for lane_def in lane_definitions:
        label = lane_def["name"]
        w = lane_def["replay_weights"]
        r = [generic_rows] if len(w) == 1 else [generic_rows, random_rows]
        shares = effective_sample_share(w, r)
        lane_def["effective_shares"] = shares
        if len(w) == 1:
            effective_shares[label] = {"generic": shares[0]}
        else:
            effective_shares[label] = {"generic": shares[0], "mined": shares[1]}

    dataset_info["lane_definitions"] = [
        {
            "lane": ld["name"],
            "replay_weights": ld["replay_weights"],
            "effective_shares": effective_shares[ld["name"]],
        }
        for ld in lane_definitions
    ]

    lane_results: dict[str, dict] = {}
    top1_artifact_dirs: dict[str, Path] = {}

    for lane_def in lane_definitions:
        name = lane_def["name"]
        print(f"\n=== Training lane: {name} ===")
        print(f"  files: {lane_def['data_files']}")
        print(f"  weights: {lane_def['replay_weights']}")
        print(f"  effective shares: {effective_shares[name]}")

        if args.skip_train:
            print(f"  [skip-train] skipping training for {name}")
            continue

        train_result = train_lane(
            python=python,
            workdir=workdir,
            lane_name=name,
            data_files=lane_def["data_files"],
            replay_weights=lane_def["replay_weights"],
            init_checkpoint=init_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_top_k=args.save_top_k,
            seed=args.seed,
            cwd=root,
            timeout=args.timeout,
        )

        if train_result["status"] != "completed":
            print(f"  TRAIN FAILED: {train_result.get('stderr', '')}")
            lane_results[name] = {"train": train_result, "status": "train_failed"}
            continue

        topk = train_result.get("topk_checkpoints", [])
        if not topk:
            print("  No top-k checkpoints found!")
            lane_results[name] = {"train": train_result, "status": "no_checkpoints"}
            continue

        top1_path = Path(topk[0]["path"])

        # Export top1 as artifact
        artifact_dir = workdir / name / "artifact"
        print(f"  Exporting top1 checkpoint to {artifact_dir}")

        export_result = export_artifact(
            python=python,
            checkpoint_path=top1_path,
            artifact_dir=artifact_dir,
            version=f"{name}-top1",
            cwd=root,
            timeout=300,
        )

        if export_result["status"] != "completed":
            print("  EXPORT FAILED")
            lane_results[name] = {
                "train": train_result,
                "export": export_result,
                "status": "export_failed",
            }
            continue

        top1_artifact_dirs[name] = artifact_dir
        lane_results[name] = {
            "train": train_result,
            "export": export_result,
            "status": "trained",
        }

    if not args.skip_arena:
        for name, result in list(lane_results.items()):
            if result["status"] != "trained":
                continue

            artifact_dir = top1_artifact_dirs[name]
            arena_dir = workdir / name / "arena"
            arena_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n=== Arena evaluation: {name} (standard {args.arena_games} games) ==="
            )

            std_report = arena_dir / "standard_arena.json"
            arena_result = run_arena(
                python=python,
                challenger_path=artifact_dir,
                current_path=str(root / "model-artifact/current"),
                games=args.arena_games,
                challenger_simulations=args.challenger_simulations,
                current_simulations=args.current_simulations,
                report_path=std_report,
                workers=args.workers,
                seed=args.seed,
                cwd=root,
                timeout=args.timeout,
            )

            lane_results[name]["standard_arena"] = arena_result
            std_arena_data = arena_result.get("arena", {})
            std_score = std_arena_data.get("score", 0.0)

            print(
                f"  standard_arena_score={std_score} "
                f"wins={std_arena_data.get('wins')} "
                f"losses={std_arena_data.get('losses')} "
                f"draws={std_arena_data.get('draws')}"
            )

            if (
                0.48 <= std_score <= 0.57
                and args.extended_arena_games > args.arena_games
            ):
                print(
                    f"  Near-threshold score ({std_score}), running extended arena "
                    f"({args.extended_arena_games} games)..."
                )
                ext_report = arena_dir / "extended_arena.json"
                ext_result = run_arena(
                    python=python,
                    challenger_path=artifact_dir,
                    current_path=str(root / "model-artifact/current"),
                    games=args.extended_arena_games,
                    challenger_simulations=args.challenger_simulations,
                    current_simulations=args.current_simulations,
                    report_path=ext_report,
                    workers=args.workers,
                    seed=args.seed + 1000,
                    cwd=root,
                    timeout=args.timeout,
                )
                lane_results[name]["extended_arena"] = ext_result
                ext_data = ext_result.get("arena", {})
                print(
                    f"  extended_arena_score={ext_data.get('score')} "
                    f"wins={ext_data.get('wins')} "
                    f"losses={ext_data.get('losses')} "
                    f"draws={ext_data.get('draws')}"
                )

    if not args.skip_gate:
        for name, result in list(lane_results.items()):
            if result["status"] != "trained":
                continue

            artifact_dir = top1_artifact_dirs[name]
            gate_dir = workdir / name / "gate"
            gate_dir.mkdir(parents=True, exist_ok=True)
            gate_out = gate_dir / "promotion.json"

            print(f"\n=== Promotion gate: {name} ===")
            gate_result = run_gate(
                python=python,
                candidate_path=artifact_dir,
                out_path=gate_out,
                current_path=str(root / "model-artifact/current"),
                arena_games=args.arena_games,
                cwd=root,
                timeout=args.timeout,
            )
            lane_results[name]["gate"] = gate_result
            gate_data = gate_result.get("gate", {})
            print(
                f"  gate_passed={gate_data.get('passed')} "
                f"failures={gate_data.get('failure_reasons')}"
            )

    # Build summary
    lanes_summary = {}
    for name in lane_results:
        lr = lane_results[name]
        train_data = lr.get("train", {})
        metrics = (
            train_data.get("metrics", {})
            if isinstance(train_data.get("metrics"), dict)
            else {}
        )
        topk = (
            train_data.get("topk_checkpoints", [])
            if isinstance(train_data.get("topk_checkpoints"), list)
            else []
        )

        std_arena = lr.get("standard_arena", {})
        std_data = (
            std_arena.get("arena", {})
            if isinstance(std_arena.get("arena"), dict)
            else {}
        )
        ext_arena = lr.get("extended_arena", {})
        ext_data = (
            ext_arena.get("arena", {})
            if isinstance(ext_arena.get("arena"), dict)
            else {}
        )
        gate = lr.get("gate", {})
        gate_data = gate.get("gate", {}) if isinstance(gate.get("gate"), dict) else {}

        ld = next((ld for ld in lane_definitions if ld["name"] == name), None)
        eff_shares = effective_shares.get(name, {})

        lanes_summary[name] = {
            "status": lr.get("status", "unknown"),
            "replay_weights": ld["replay_weights"] if ld else [],
            "effective_shares": eff_shares,
            "training": {
                "policy_loss": metrics.get("policy_loss"),
                "value_loss": metrics.get("value_loss"),
                "best_val_loss": metrics.get("best_val_loss"),
                "top1_sha256": topk[0]["sha256"] if topk else None,
            },
            "standard_arena": {
                "score": std_data.get("score"),
                "games_played": std_data.get("games_played"),
                "wins": std_data.get("wins"),
                "losses": std_data.get("losses"),
                "draws": std_data.get("draws"),
                "ci95_low": std_data.get("ci95_low"),
                "ci95_high": std_data.get("ci95_high"),
                "move_time_mean_ms": std_data.get("move_time_mean_ms"),
                "move_time_p95_ms": std_data.get("move_time_p95_ms"),
            },
            "extended_arena": {
                "score": ext_data.get("score"),
                "games_played": ext_data.get("games_played"),
                "wins": ext_data.get("wins"),
                "losses": ext_data.get("losses"),
                "draws": ext_data.get("draws"),
                "ci95_low": ext_data.get("ci95_low"),
                "ci95_high": ext_data.get("ci95_high"),
            }
            if ext_data
            else None,
            "gate": {
                "passed": gate_data.get("passed"),
                "arena_score": gate_data.get("arena_score"),
                "failure_reasons": gate_data.get("failure_reasons", []),
            },
        }

    full_summary = {
        "schema": "random_replay_weight_sweep_v1",
        "workdir": str(workdir),
        "dataset": dataset_info,
        "lanes": lanes_summary,
    }

    summary_path = workdir / "sweep_summary.json"
    write_json(summary_path, full_summary)
    print(f"\nWrote sweep summary to {summary_path}")

    # Generate markdown report
    md_path = workdir / "sweep_report.md"
    md_lines = _render_markdown(full_summary, args)
    md_path.write_text(md_lines, encoding="utf-8")
    print(f"Wrote markdown report to {md_path}")

    docs_path = (
        repo_root() / "docs/alphazero-lite-random-replay-weight-sweep-results.md"
    )
    docs_path.write_text(md_lines, encoding="utf-8")
    print(f"Wrote docs to {docs_path}")

    return 0


def _render_markdown(summary: dict, args: argparse.Namespace) -> str:
    ds = summary["dataset"]
    lanes = summary["lanes"]
    gi = ds["generic_replay"]
    ri = ds["random_mined_replay"]
    init = ds["init_checkpoint"]

    lines = [
        "# AlphaZero-Lite Random Replay Weight Sweep Results",
        "",
        "**Date:** 2026-06-07",
        "",
        "## Classification",
        "",
    ]

    baseline_score = None
    pr91_score = 0.50

    for name, lane in lanes.items():
        if lane.get("replay_weights") == [4]:
            baseline_score = lane.get("standard_arena", {}).get("score")
            break

    random_results = []
    for name, lane in lanes.items():
        if len(lane.get("replay_weights", [])) == 2:
            std = lane.get("standard_arena") or {}
            ext = lane.get("extended_arena")
            score = std.get("score", 0.0)
            eff = (lane.get("effective_shares") or {}).get("mined", 0.0)
            random_results.append((name, score, eff, std, ext, lane))

    random_results.sort(key=lambda x: x[2], reverse=True)

    best_name, best_score, best_eff, _, _, _ = (
        random_results[0] if random_results else ("none", 0.0, 0.0, {}, None, {})
    )

    if best_score >= 0.55:
        lines.append(
            "**PROMISING** — at least one random-weight lane reached or exceeded the 0.55 local promotion arena threshold."
        )
    elif best_score > pr91_score:
        lines.append(
            f"**PROMISING** — at least one random-weight lane (score={best_score:.2f}) beats the PR #91 random score ({pr91_score})."
        )
    elif best_score == pr91_score:
        lines.append(
            f"**CAPPED** — all random-weight lanes cluster at or below {pr91_score}. Higher mined effective sample share does not improve arena strength."
        )
    else:
        lines.append(
            "**BELOW BASELINE** — random-weight lanes fell below the PR #91 random score."
        )

    lines += [
        "",
        "## Experiment Design",
        "",
    ]

    lines.append(
        "| Lane | Replay Weights | Effective Generic Share | Effective Mined Share |"
    )
    lines.append(
        "|------|---------------|------------------------|----------------------|"
    )
    for ld in ds.get("lane_definitions", []):
        eff = ld.get("effective_shares", {})
        gen = eff.get("generic", 0)
        mined = eff.get("mined", 0)
        lines.append(
            f"| {ld['lane']} | {','.join(str(w) for w in ld['replay_weights'])} | "
            f"{gen:.1%} | {mined:.1%} |"
        )

    lines += [
        "",
        "## Dataset",
        "",
    ]
    lines.append("### Generic Replay")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Path | `{gi['path']}` |")
    lines.append(f"| Rows | {gi['rows']:,} |")
    lines.append(f"| SHA256 | `{gi['sha256']}` |")
    lines.append("")
    lines.append("### Random Mined Replay")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Path | `{ri['path']}` |")
    lines.append(f"| Rows | {ri['rows']:,} |")
    lines.append(f"| SHA256 | `{ri['sha256']}` |")
    lines.append("")
    lines.append("### Init Checkpoint")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Source | `{init['source']}` |")
    lines.append(f"| SHA256 | `{init['sha256']}` |")

    lines += [
        "",
        "## Training",
        "",
    ]

    lines.append("| Metric | " + " | ".join(lanes.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(lanes)) + "|")

    metric_keys = [
        ("Policy Loss", lambda la: (la.get("training") or {}).get("policy_loss")),
        ("Value Loss", lambda la: (la.get("training") or {}).get("value_loss")),
        ("Best Val Loss", lambda la: (la.get("training") or {}).get("best_val_loss")),
        ("Status", lambda la: la.get("status")),
    ]

    for label, getter in metric_keys:
        row = [label]
        for name in lanes:
            val = getter(lanes[name])
            if val is None:
                row.append("—")
            elif isinstance(val, float):
                row.append(f"{val:.6f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Strength: Standard Arena vs Current",
        "",
    ]

    lines.append("| Metric | " + " | ".join(lanes.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(lanes)) + "|")

    arena_metrics = [
        ("Arena Score", lambda la: (la.get("standard_arena") or {}).get("score")),
        ("Games", lambda la: (la.get("standard_arena") or {}).get("games_played")),
        ("Wins", lambda la: (la.get("standard_arena") or {}).get("wins")),
        ("Losses", lambda la: (la.get("standard_arena") or {}).get("losses")),
        ("Draws", lambda la: (la.get("standard_arena") or {}).get("draws")),
        ("CI95 Low", lambda la: (la.get("standard_arena") or {}).get("ci95_low")),
        ("CI95 High", lambda la: (la.get("standard_arena") or {}).get("ci95_high")),
        (
            "Move Time Mean ms",
            lambda la: (la.get("standard_arena") or {}).get("move_time_mean_ms"),
        ),
        (
            "Move Time P95 ms",
            lambda la: (la.get("standard_arena") or {}).get("move_time_p95_ms"),
        ),
    ]

    for label, getter in arena_metrics:
        row = [label]
        for name in lanes:
            val = getter(lanes[name])
            if val is None:
                row.append("—")
            elif isinstance(val, float):
                if "ci95" in label.lower():
                    row.append(f"{val:.4f}")
                elif "score" in label.lower():
                    row.append(f"{val:.4f}")
                else:
                    row.append(f"{val:.2f}" if "ms" in label.lower() else str(val))
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    # Extended arena
    has_extended = any(
        (lanes[n].get("extended_arena") or {}).get("score") is not None for n in lanes
    )
    if has_extended:
        lines += [
            "",
            "## Strength: Extended Arena vs Current",
            "",
        ]
        lines.append("| Metric | " + " | ".join(lanes.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(lanes)) + "|")
        for label, getter in [
            ("Score", lambda la: (la.get("extended_arena") or {}).get("score")),
            ("Games", lambda la: (la.get("extended_arena") or {}).get("games_played")),
            ("Wins", lambda la: (la.get("extended_arena") or {}).get("wins")),
            ("Losses", lambda la: (la.get("extended_arena") or {}).get("losses")),
            ("Draws", lambda la: (la.get("extended_arena") or {}).get("draws")),
            ("CI95 Low", lambda la: (la.get("extended_arena") or {}).get("ci95_low")),
            ("CI95 High", lambda la: (la.get("extended_arena") or {}).get("ci95_high")),
        ]:
            row = [label]
            for name in lanes:
                val = getter(lanes[name])
                if val is None:
                    row.append("—")
                elif isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            lines.append("| " + " | ".join(row) + " |")

    # Gate results
    lines += [
        "",
        "## Promotion Gate",
        "",
    ]
    lines.append("| Metric | " + " | ".join(lanes.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(lanes)) + "|")
    for label, getter in [
        ("Passed", lambda la: (la.get("gate") or {}).get("passed")),
        ("Score", lambda la: (la.get("gate") or {}).get("arena_score")),
        (
            "Failures",
            lambda la: (
                ", ".join(
                    str(f.get("code", f))
                    for f in (la.get("gate") or {}).get("failure_reasons", [])
                )
                or "—"
            ),
        ),
    ]:
        row = [label]
        for name in lanes:
            val = getter(lanes[name])
            if val is None:
                row.append("—")
            elif isinstance(val, bool):
                row.append("Yes" if val else "No")
            elif isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Acceptance Criteria Evaluation",
        "",
    ]

    for name, score, eff, std, ext, lane in random_results:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Criterion | Result | Status |")
        lines.append("|-----------|--------|--------|")
        lines.append(
            f"| Reach 0.55 threshold | {score:.4f} | "
            f"{'PASS' if score >= 0.55 else 'FAIL'} |"
        )
        lines.append(
            f"| Beat PR #91 random (0.50) | {score:.4f} | "
            f"{'PASS' if score > 0.50 else 'FAIL' if score < 0.50 else 'TIE'} |"
        )
        if ext and ext.get("score") is not None:
            lines.append(
                f"| Extended arena score | {ext['score']:.4f} | "
                f"{'PASS' if ext['score'] > 0.50 else 'FAIL' if ext['score'] < 0.50 else 'TIE'} |"
            )
        lines.append("")

    if baseline_score is not None:
        lines.append("### Baseline comparison")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Baseline arena score | {baseline_score:.4f} |")
        lines.append(f"| PR #91 random score | {pr91_score} |")
        for name, score, eff, _, _, _ in random_results:
            lines.append(f"| {name} score | {score:.4f} (eff share={eff:.1%}) |")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
