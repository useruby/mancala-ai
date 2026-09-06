#!/usr/bin/env python3
"""Controlled exact-vs-MCTS training ablation on the frozen 10k cohort.

Trains two matched lanes on the IDENTICAL 8,000 frozen train states:

* ``exact``  lane: native-hybrid exact labels (uniform optimal policy,
  sign-of-margin value);
* ``mcts``   lane: classic-MCTS 1200-simulation labels for the same states.

Architecture, optimizer, seed, epochs, batch size, learning rate, and all
other hyperparameters are fixed across lanes. Only the teacher targets
differ. Both lanes share the same ``--val-split`` seed path, so internal
validation partitions match row-for-row.

Evaluation (no promotion):

* frozen 2,000-state exact holdout scored with exact-optimal-set metrics
  (top-1 in optimal set, single-optimum agreement, value MAE);
* sealed arena: exact-challenger vs mcts-challenger head-to-head plus a
  shared current-control leg for the paired candidate-effect statistic.

This runner never promotes. It writes checkpoints, arena records, and a
machine-readable summary plus a Markdown report section payload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import exact_teacher_labeling as exact  # noqa: E402
from ml.alphazero_lite.arena import run_arena_worker  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (  # noqa: E402
    decode_kalah_v3_base_state,
    state_round_trips_kalah_v3,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    sha256_file,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    apply_trainable_scope,
    checkpoint_from_state_dict,
    input_size_for_encoding,
    set_seed,
)

LANES = ("exact", "mcts")
DEFAULT_MODEL_TYPE = "mlp_v1"
DEFAULT_HIDDEN_SIZES = "64,64"
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
DEFAULT_VALUE_LOSS_WEIGHT = 0.5
ARENA_CONTEXT = "384:256"
ARENA_WORKERS = 8
ARENA_GAMES_PER_OPENING = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-train", type=Path, required=True)
    parser.add_argument("--mcts-train", type=Path, required=True)
    parser.add_argument("--exact-holdout", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--hidden-sizes", default=DEFAULT_HIDDEN_SIZES)
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--current", type=Path, default=Path("storage/ai/alphazero_lite/current")
    )
    parser.add_argument(
        "--arena-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--arena-context", default=ARENA_CONTEXT)
    parser.add_argument("--arena-workers", type=int, default=ARENA_WORKERS)
    parser.add_argument(
        "--arena-games-per-opening", type=int, default=ARENA_GAMES_PER_OPENING
    )
    parser.add_argument("--source-states", type=Path, default=None)
    parser.add_argument("--skip-arena", action="store_true")
    return parser.parse_args(argv)


def verify_identical_states(exact_rows: list[dict], mcts_rows: list[dict]) -> None:
    if len(exact_rows) != len(mcts_rows):
        raise ValueError(
            f"lane row-count mismatch: exact={len(exact_rows)} mcts={len(mcts_rows)}"
        )
    for index, (left, right) in enumerate(zip(exact_rows, mcts_rows)):
        if left["source_id"] != right["source_id"]:
            raise ValueError(
                f"row {index}: source_id drift {left['source_id']} vs {right['source_id']}"
            )
        if left["canonical_state"] != right["canonical_state"]:
            raise ValueError(f"row {index}: canonical-state drift")
        if not np.allclose(np.asarray(left["state"]), np.asarray(right["state"])):
            raise ValueError(f"row {index}: encoded-state drift")


def verify_holdout_disjoint(train_rows: list[dict], holdout_rows: list[dict]) -> None:
    train_keys = {r["canonical_state"] for r in train_rows}
    holdout_keys = {r["canonical_state"] for r in holdout_rows}
    if train_keys & holdout_keys:
        raise ValueError("train/holdout canonical overlap in ablation inputs")
    if not holdout_keys:
        raise ValueError("holdout is empty")


def load_lane_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([r["state"] for r in rows], dtype=np.float32)
    p = np.asarray([r["policy"] for r in rows], dtype=np.float32)
    v = np.asarray([r["value"] for r in rows], dtype=np.float32).reshape(-1, 1)
    return x, p, v


def train_lane(
    rows: list[dict],
    *,
    model_type: str,
    hidden_sizes: tuple[int, ...],
    input_encoding: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    value_loss_weight: float,
    val_split: float,
    device: torch.device,
    lane_dir: Path,
) -> dict[str, Any]:
    from ml.alphazero_lite.train import train as train_fn

    lane_dir.mkdir(parents=True, exist_ok=True)
    configure_determinism(device, seed)
    set_seed(seed)
    model = PolicyValueNet(
        hidden_sizes=hidden_sizes,
        model_type=model_type,
        input_size=input_size_for_encoding(input_encoding),
    )
    apply_trainable_scope(model, "all")
    x, p, v = load_lane_arrays(rows)
    policy_loss, value_loss, best_val = train_fn(
        model,
        x,
        p,
        v,
        None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        value_loss_weight=value_loss_weight,
        value_loss="huber",
        huber_delta=1.0,
        val_split=val_split,
        grad_clip=1.0,
        save_top_k=0,
    )
    checkpoint = lane_dir / "checkpoint.npz"
    arrays = {
        k: np.asarray(v_)
        for k, v_ in checkpoint_from_state_dict(model.state_dict()).items()
    }
    np.savez(checkpoint, **arrays)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "best_val_loss": float(best_val),
    }


def export_artifact(
    checkpoint: Path,
    out_dir: Path,
    *,
    model_type: str,
    input_encoding: str,
    version: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
            "--version",
            version,
            "--model-type",
            model_type,
            "--input-encoding",
            input_encoding,
            "--rules-version",
            "kalah_v1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return out_dir


class _NpzEvaluator:
    """Minimal numpy evaluator mirroring CheckpointEvaluator for scoring."""

    def __init__(self, checkpoint: Path, *, input_encoding: str) -> None:
        from ml.alphazero_lite.self_play import CheckpointEvaluator  # noqa: E402

        self._impl = CheckpointEvaluator(checkpoint, input_encoding=input_encoding)

    def evaluate(self, game):  # type: ignore[no-untyped-def]
        return self._impl.evaluate(game)


def score_holdout(
    checkpoint: Path,
    holdout_rows: list[dict],
    *,
    input_encoding: str,
    bucket_by_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import top_policy_move_for_legal_moves

    evaluator = _NpzEvaluator(checkpoint, input_encoding=input_encoding)
    top_in_set = 0
    single_total = 0
    single_agree = 0
    multi_total = 0
    abs_value_errors: list[float] = []
    policy_ce: list[float] = []
    by_bucket: dict[str, dict[str, Any]] = {}
    for row in holdout_rows:
        state = decode_kalah_v3_base_state(list(row["state"]))
        if not state_round_trips_kalah_v3(list(row["state"])):
            raise ValueError(f"holdout state fails v3 round-trip: {row['source_id']}")
        game = KalahGame.from_state(state)
        legal = game.possible_moves()
        policy_logits, value = evaluator.evaluate(game)
        policy_logits = np.asarray(policy_logits, dtype=np.float64)
        mask = np.zeros(6)
        mask[legal] = 1.0
        policy_logits = np.where(mask > 0, policy_logits, -1e9)
        policy_logits -= policy_logits.max()
        probs = np.exp(policy_logits)
        probs /= probs.sum()
        optimal = [int(a) for a in row["exact_optimal_actions"]]
        top = int(top_policy_move_for_legal_moves(list(probs), legal))
        if top in set(optimal):
            top_in_set += 1
        if len(optimal) == 1:
            single_total += 1
            single_agree += int(top == optimal[0])
        else:
            multi_total += 1
        abs_value_errors.append(abs(float(value) - float(row["exact_value_training"])))
        target = np.zeros(6)
        target[optimal] = 1.0 / len(optimal)
        policy_ce.append(float(-np.sum(target[legal] * np.log(probs[legal] + 1e-12))))
        bucket_rows = by_bucket.setdefault(
            source_bucket(row, bucket_by_source),
            {"n": 0, "top_in_set": 0, "abs_err": 0.0},
        )
        bucket_rows["n"] += 1
        bucket_rows["top_in_set"] += int(top in set(optimal))
        bucket_rows["abs_err"] += abs(float(value) - float(row["exact_value_training"]))
    total = len(holdout_rows)
    summary = {
        "n": total,
        "top_in_exact_set": top_in_set,
        "top_in_exact_set_rate": top_in_set / total,
        "single_n": single_total,
        "single_top1_agreement": (single_agree / single_total)
        if single_total
        else None,
        "multi_n": multi_total,
        "multi_rate": multi_total / total,
        "value_mae": float(np.mean(abs_value_errors)),
        "value_max_ae": float(np.max(abs_value_errors)),
        "policy_cross_entropy_vs_uniform_optimal": float(np.mean(policy_ce)),
        "by_bucket": {
            bucket: {
                "n": info["n"],
                "top_in_exact_set_rate": info["top_in_set"] / info["n"],
                "value_mae": info["abs_err"] / info["n"],
            }
            for bucket, info in sorted(by_bucket.items())
        },
    }
    return summary


def source_bucket(
    row: dict[str, Any], bucket_by_source: dict[str, str] | None = None
) -> str:
    if isinstance(bucket_by_source, dict):
        bucket = bucket_by_source.get(str(row.get("source_id", "")))
        if bucket:
            return str(bucket)
    return str(row.get("stones_bucket", row.get("bucket", "unknown")))


def run_arena_leg(
    workdir: Path,
    challenger: Path,
    current: Path,
    suite: Path,
    *,
    context: str,
    workers: int,
    games_per_opening: int,
    role: str,
) -> list[dict[str, Any]]:
    challenger_sims, current_sims = (int(v) for v in context.split(":"))
    records: list[dict[str, Any]] = []
    for seat in (0, 1):
        total = 256
        counts = [total // workers + (i < total % workers) for i in range(workers)]
        starts: list[int] = []
        cursor = 0
        for count in counts:
            starts.append(cursor)
            cursor += count
        kwargs: dict[str, Any] = {
            "challenger_path": str(challenger),
            "current_path": str(current),
            "challenger_simulations": challenger_sims,
            "current_simulations": current_sims,
            "seed": 42,
            "c_puct": 1.25,
            "max_moves": 200,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
            "games_per_opening": games_per_opening,
            "challenger_starts": seat,
            "opening_prefixes_jsonl": str(suite),
            "suite_sha256_override": sha256_file(suite),
        }
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_arena_worker, worker_id=i, start_index=s, games=c, **kwargs
                )
                for i, (s, c) in enumerate(zip(starts, counts))
                if c
            ]
            results = [f.result() for f in futures]
        for result in results:
            for game_row in result["game_entries"]:
                game_row["opponent_weights_sha256"] = sha256_file(
                    current / "weights.json"
                )
                game_row["opponent_config_sha256"] = f"{context}:1.25"
            records.extend(result["game_entries"])
    (workdir / role).mkdir(parents=True, exist_ok=True)
    (workdir / role / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def bucket_map_for_holdout(args: argparse.Namespace) -> dict[str, str]:
    source_states = getattr(args, "source_states", None)
    if not source_states:
        return {}
    path = Path(str(source_states))
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    for row in exact.read_jsonl(path):
        bucket = row.get("stones_bucket")
        if bucket:
            mapping[str(row.get("source_id", ""))] = str(bucket)
    return mapping


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    exact_rows = exact.read_jsonl(args.exact_train)
    mcts_rows = exact.read_jsonl(args.mcts_train)
    holdout_rows = exact.read_jsonl(args.exact_holdout)
    verify_identical_states(exact_rows, mcts_rows)
    verify_holdout_disjoint(exact_rows, holdout_rows)
    bucket_by_source = bucket_map_for_holdout(args)
    hidden_sizes = tuple(int(v) for v in str(args.hidden_sizes).split(","))
    device = torch.device(args.device)
    config = {
        "model_type": args.model_type,
        "hidden_sizes": list(hidden_sizes),
        "input_encoding": args.input_encoding,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "value_loss_weight": float(args.value_loss_weight),
        "val_split": float(args.val_split),
    }
    results: dict[str, Any] = {"config": config, "lanes": {}}
    for lane, rows in (("exact", exact_rows), ("mcts", mcts_rows)):
        lane_result = train_lane(
            rows,
            model_type=args.model_type,
            hidden_sizes=hidden_sizes,
            input_encoding=args.input_encoding,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            value_loss_weight=float(args.value_loss_weight),
            val_split=float(args.val_split),
            device=device,
            lane_dir=args.workdir / lane,
        )
        artifact = export_artifact(
            Path(lane_result["checkpoint"]),
            args.workdir / lane / "artifact",
            model_type=args.model_type,
            input_encoding=args.input_encoding,
            version=f"exact-ablation-{lane}",
        )
        lane_result["artifact"] = str(artifact)
        lane_result["holdout"] = score_holdout(
            Path(lane_result["checkpoint"]),
            holdout_rows,
            input_encoding=args.input_encoding,
            bucket_by_source=bucket_by_source,
        )
        results["lanes"][lane] = lane_result
        print(
            f"lane={lane} policy_loss={lane_result['policy_loss']:.4f} "
            f"value_loss={lane_result['value_loss']:.4f} "
            f"holdout_top_in_set={lane_result['holdout']['top_in_exact_set_rate']:.4f} "
            f"holdout_value_mae={lane_result['holdout']['value_mae']:.4f}",
            flush=True,
        )
    exact_holdout = results["lanes"]["exact"]["holdout"]
    mcts_holdout = results["lanes"]["mcts"]["holdout"]
    results["holdout_contrast"] = {
        "top_in_set_rate_diff_exact_minus_mcts": (
            exact_holdout["top_in_exact_set_rate"]
            - mcts_holdout["top_in_exact_set_rate"]
        ),
        "value_mae_diff_exact_minus_mcts": (
            exact_holdout["value_mae"] - mcts_holdout["value_mae"]
        ),
    }
    if not args.skip_arena:
        suite = args.arena_suite
        current = args.current
        if not suite.is_file():
            raise SystemExit(f"arena suite not found: {suite}")
        if not (current / "weights.json").is_file():
            raise SystemExit(f"current artifact weights not found: {current}")
        exact_records = run_arena_leg(
            args.workdir / "arena",
            Path(results["lanes"]["exact"]["artifact"]),
            current,
            suite,
            context=args.arena_context,
            workers=args.arena_workers,
            games_per_opening=args.arena_games_per_opening,
            role="exact_challenger",
        )
        mcts_records = run_arena_leg(
            args.workdir / "arena",
            Path(results["lanes"]["mcts"]["artifact"]),
            current,
            suite,
            context=args.arena_context,
            workers=args.arena_workers,
            games_per_opening=args.arena_games_per_opening,
            role="mcts_challenger",
        )
        # Head-to-head: exact-challenger vs mcts-challenger as current.
        h2h = run_arena_leg(
            args.workdir / "arena",
            Path(results["lanes"]["exact"]["artifact"]),
            Path(results["lanes"]["mcts"]["artifact"]),
            suite,
            context=args.arena_context,
            workers=args.arena_workers,
            games_per_opening=args.arena_games_per_opening,
            role="exact_vs_mcts",
        )
        results["arena"] = {
            "suite": str(suite),
            "suite_sha256": sha256_file(suite),
            "context": args.arena_context,
            "games_per_opening": int(args.arena_games_per_opening),
            "exact_vs_shared_current_effect": paired_opening_candidate_effect(
                exact_records, mcts_records
            ),
            "head_to_head_exact_vs_mcts": {
                "games": len(h2h),
                "exact_score": float(np.mean([score_of(r) for r in h2h])),
            },
        }
    else:
        results["arena"] = {"skipped": True}
    results.update(
        {
            "schema": "exact_teacher_training_ablation_v1",
            "exact_train": str(args.exact_train),
            "mcts_train": str(args.mcts_train),
            "exact_holdout": str(args.exact_holdout),
            "exact_train_sha256": sha256_file(args.exact_train),
            "mcts_train_sha256": sha256_file(args.mcts_train),
            "holdout_sha256": sha256_file(args.exact_holdout),
            "identical_state_vectors": True,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "provenance": {
                "torch_version": torch.__version__,
                "train_seed": int(args.seed),
            },
        }
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ablation_written={args.out_summary}")
    return 0


def score_of(record: dict[str, Any]) -> float:
    winner = record.get("winner")
    if winner == "challenger":
        return 1.0
    if winner == "draw":
        return 0.5
    if winner == "current":
        return 0.0
    return float(record.get("score", 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
