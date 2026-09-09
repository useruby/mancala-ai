#!/usr/bin/env python3
"""Run PR #287's row-count-normalized uniform1200 replay ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.self_play import value_target_bucket_for_move_index  # noqa: E402

SCHEMA = "azlite_uniform1200_rowmatched_v1"
LANE = "uniform1200_rowmatched"
PHASE_TOLERANCE_PERCENTAGE_POINTS = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        value_target_bucket_for_move_index(int(row.get("move_index", 0))),
        str(row.get("player")),
        str(row.get("winner")),
    )


def stable_row_digest(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def proportional_stratified_sample(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select exactly count rows while preserving phase/player/outcome proportions."""
    if not 0 <= count <= len(rows):
        raise ValueError("requested sample count is outside the source row count")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_key(row)].append(row)
    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key, group in groups.items():
        exact = len(group) * count / len(rows)
        quotas[key] = math.floor(exact)
        remainders.append((exact - quotas[key], key))
    for _remainder, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : count - sum(quotas.values())
    ]:
        quotas[key] += 1
    sampled = []
    for key in sorted(groups):
        sampled.extend(sorted(groups[key], key=stable_row_digest)[: quotas[key]])
    if len(sampled) != count:
        raise AssertionError(
            "proportional allocation did not produce the requested count"
        )
    return sampled


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def percentages(values: list[str]) -> dict[str, float]:
        counts = Counter(values)
        return {
            key: round(100 * value / len(rows), 4)
            for key, value in sorted(counts.items())
        }

    entropies = []
    starts = []
    states = []
    for row in rows:
        policy = [
            float(value) for value in row.get("policy", row.get("policy_target", []))
        ]
        entropies.append(
            -sum(value * math.log2(value) for value in policy if value > 0)
        )
        states.append(json.dumps(row.get("state"), separators=(",", ":")))
        if int(row.get("move_index", -1)) == 0:
            starts.append(states[-1])
    return {
        "phase_percent": percentages([row_key(row)[0] for row in rows]),
        "player_percent": percentages([row_key(row)[1] for row in rows]),
        "outcome_percent": percentages([row_key(row)[2] for row in rows]),
        "policy_entropy": {
            "mean": round(sum(entropies) / len(entropies), 6),
            "min": round(min(entropies), 6),
            "max": round(max(entropies), 6),
        },
        "trajectory_starts": len(starts),
        "unique_trajectory_starts": len(set(starts)),
        "canonical_unique_states": len(set(states)),
    }


def check_distribution(full: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    deviations = {}
    for dimension in ("phase_percent", "player_percent", "outcome_percent"):
        deviations[dimension] = {
            key: round(sampled[dimension].get(key, 0.0) - value, 4)
            for key, value in full[dimension].items()
        }
    violations = [
        f"{dimension}:{key}"
        for dimension, values in deviations.items()
        for key, delta in values.items()
        if abs(delta) > PHASE_TOLERANCE_PERCENTAGE_POINTS
    ]
    return {
        "tolerance_percentage_points": PHASE_TOLERANCE_PERCENTAGE_POINTS,
        "deviations": deviations,
        "violations": violations,
        "passed": not violations,
    }


def artifact_dir(root: Path, seed: int, lane: str) -> Path:
    return (
        root
        / "runs"
        / f"seed{seed}"
        / lane
        / f"uniform1200-confirm-seed{seed}-{lane}-iter1"
    )


def exposure(sources: list[tuple[str, Path, int]]) -> dict[str, Any]:
    entries = [
        {"source": name, "rows": len(load_rows(path)), "weight": weight}
        for name, path, weight in sources
    ]
    total = sum(entry["rows"] * entry["weight"] for entry in entries)
    for entry in entries:
        entry["effective_sampled_index_count"] = entry["rows"] * entry["weight"]
        entry["fraction"] = entry["effective_sampled_index_count"] / total
    return {"sources": entries, "total_effective_sampled_index_count": total}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("ml/alphazero_lite/configs/uniform1200_rowmatched.json"),
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA or plan["promotion"].get("performed") is not False:
        raise ValueError("invalid or promotable row-matched plan")
    root = Path(plan["artifact_root"])
    selected, controls = Path(plan["selected_replay"]), Path(plan["controls_replay"])
    expected = plan["expected"]
    parent = REPO_ROOT / "storage/ai/alphazero_lite/current/weights.json"
    for path, digest, name in (
        (parent, expected["parent_weights_sha256"], "parent"),
        (selected, expected["selected_replay_sha256"], "selected replay"),
        (controls, expected["controls_replay_sha256"], "controls replay"),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"PR #287 {name} SHA-256 mismatch")
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "promotion": plan["promotion"],
        "seeds": {},
    }
    for seed in plan["seeds"]:
        control_dir, uniform_dir = (
            artifact_dir(root, seed, "control_384_192"),
            artifact_dir(root, seed, "uniform1200"),
        )
        control_replay, uniform_replay = (
            control_dir / "self_play.jsonl",
            uniform_dir / "self_play.jsonl",
        )
        for path in (
            control_replay,
            uniform_replay,
            control_dir / "checkpoint.npz",
            uniform_dir / "checkpoint.npz",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing locked PR #287 artifact: {path}")
        # PR #287 records the exported model.npz digest as its checkpoint SHA.
        if (
            sha256_file(control_dir / "model.npz")
            != expected["checkpoints"][str(seed)]["control"]
            or sha256_file(uniform_dir / "model.npz")
            != expected["checkpoints"][str(seed)]["uniform"]
        ):
            raise ValueError(f"seed {seed} checkpoint SHA-256 mismatch")
        rows = load_rows(uniform_replay)
        sampled = proportional_stratified_sample(
            rows, int(plan["target_rows"][str(seed)])
        )
        if len(sampled) != len(load_rows(control_replay)):
            raise ValueError(f"seed {seed} row match failed")
        replay_path = (
            args.workdir / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl"
        )
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in sampled
            ),
            encoding="utf-8",
        )
        full_dist, sample_dist = distribution(rows), distribution(sampled)
        comparison = check_distribution(full_dist, sample_dist)
        if not comparison["passed"]:
            raise ValueError(
                f"seed {seed} distribution tolerance failed: {comparison['violations']}"
            )
        control_exposure = exposure(
            [
                ("dynamic_current_self_play", control_replay, 1),
                ("selected_fixed_replay", selected, 1),
                ("guard_control_fixed_replay", controls, 2),
            ]
        )
        uniform_exposure = exposure(
            [
                ("dynamic_current_self_play", uniform_replay, 1),
                ("selected_fixed_replay", selected, 1),
                ("guard_control_fixed_replay", controls, 2),
            ]
        )
        matched_exposure = exposure(
            [
                ("dynamic_current_self_play", replay_path, 1),
                ("selected_fixed_replay", selected, 1),
                ("guard_control_fixed_replay", controls, 2),
            ]
        )
        if (
            control_exposure["sources"][0]["fraction"]
            != matched_exposure["sources"][0]["fraction"]
        ):
            raise ValueError(
                f"seed {seed} effective replay mixture differs from control"
            )
        record = {
            "artifacts": {
                "control_replay_sha256": sha256_file(control_replay),
                "uniform_replay_sha256": sha256_file(uniform_replay),
                "rowmatched_replay_sha256": sha256_file(replay_path),
            },
            "exposure": {
                "control": control_exposure,
                "uniform1200": uniform_exposure,
                "uniform1200_rowmatched": matched_exposure,
            },
            "distribution": {
                "full_uniform1200": full_dist,
                "rowmatched": sample_dist,
                "comparison": comparison,
            },
            "training": {"status": "not_requested", "replay": str(replay_path)},
        }
        if args.train:
            candidate_dir = (
                args.workdir
                / "runs"
                / f"seed{seed}"
                / LANE
                / f"uniform1200-rowmatched-seed{seed}-iter1"
            )
            candidate_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(REPO_ROOT / ".venv/bin/python"),
                "ml/alphazero_lite/train.py",
                "--data",
                str(replay_path),
                "--data-files",
                f"{replay_path},{selected},{controls}",
                "--replay-weights",
                "1,1,2",
                "--init-checkpoint",
                str(control_dir / "parent_init_checkpoint.npz"),
                "--out",
                str(candidate_dir / "checkpoint.npz"),
                "--epochs",
                "4",
                "--batch-size",
                "512",
                "--device",
                "auto",
                "--lr-scheduler",
                "none",
                "--hidden-sizes",
                "96,3",
                "--model-type",
                "residual_v3",
                "--input-encoding",
                "kalah_v3",
                "--value-loss",
                "huber",
                "--huber-delta",
                "1.0",
                "--value-loss-weight",
                "0.3",
                "--val-split",
                "0.1",
                "--grad-clip",
                "1.0",
                "--save-top-k",
                "3",
                "--policy-target-mode",
                "sharpened",
                "--value-target-mode",
                "sharpened",
                "--seed",
                str(seed),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            subprocess.run(
                [
                    str(REPO_ROOT / ".venv/bin/python"),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(candidate_dir / "checkpoint.npz"),
                    "--out-dir",
                    str(candidate_dir),
                    "--version",
                    candidate_dir.name,
                    "--model-type",
                    "residual_v3",
                    "--rules-version",
                    "kalah_v1",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            record["training"] = {
                "status": "completed",
                "candidate": str(candidate_dir),
                "weights_sha256": sha256_file(candidate_dir / "weights.json"),
                "command": command,
            }
        summary["seeds"][str(seed)] = record
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
