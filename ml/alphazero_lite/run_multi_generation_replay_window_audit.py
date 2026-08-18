#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit whether a larger/multi-generation replay window stabilizes the joint-trunk update.

This is the follow-up to the ``replay_sampling_instability_confirmed``
classification. It reuses existing incumbent-model self-play replays (no new
self-play), leaves targets, architecture, LR, optimizer, loss weights, and
c_puct unchanged, and re-applies the exact whole-game shard + gradient + Adam
methodology from ``run_game_shard_gradient_stability_audit.py``.

The only scientific variable is the replay window:
  W1 = PR #191 single generation (baseline);
  W2 = W1 + pilot opening-seeded generation;
  W3 = W2 + pilot standard-start generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import stable_seed
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import (
    SHARDS,
    fresh_state,
    new_model,
    pairwise,
    real_candidate,
    state_hash,
    vectors,
    virtual_vectors,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch

SOURCES = (
    ("pr191", "/tmp/azlite_shared_trunk_learning/replay.jsonl"),
    (
        "pilot_opening_seeded",
        "/tmp/azlite_distribution_aligned_selfplay/pilot_opening_seeded_replay.jsonl",
    ),
    (
        "pilot_standard",
        "/tmp/azlite_distribution_aligned_selfplay/pilot_standard_replay.jsonl",
    ),
)
CURRENT_HASH = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
BATCH_SEED_NAMESPACE = "azlite_multi_generation_window_batches_v1"


def load_source(name: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        row["_source"] = name
        rows.append(row)
    return rows


def game_identity(row: dict[str, Any]) -> str:
    return f"{row['_source']}:{row['game_index']}:{row.get('trajectory_hash', '')}"


def partition(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    games: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        games[game_identity(row)].append(index)
    assigned: dict[str, list[int]] = {shard: [] for shard in SHARDS}
    loads = {shard: 0 for shard in SHARDS}
    for identity, indexes in sorted(
        games.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        shard = min(SHARDS, key=lambda name: (loads[name], name))
        assigned[shard].extend(indexes)
        loads[shard] += len(indexes)
    return assigned


def deterministic_batches(indexes: list[int], shard: str) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed(BATCH_SEED_NAMESPACE, shard, 191))
    return [
        rng.choice(np.asarray(indexes), size=512, replace=len(indexes) < 512)
        for _ in range(32)
    ]


def median_joint_trunk_cosine(pairwise_result: dict[str, Any]) -> float:
    values = [
        item["median"] for item in pairwise_result["shared_trunk"]["joint"].values()
    ]
    return float(np.median(values))


def window_summary(
    rows: list[dict[str, Any]],
    assignments: dict[str, list[int]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }
    raw: dict[str, list[dict[str, dict[str, torch.Tensor]]]] = {
        name: [] for name in SHARDS
    }
    updates: dict[str, list[dict[str, dict[str, torch.Tensor]]]] = {
        name: [] for name in SHARDS
    }
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    for name in SHARDS:
        for batch in batches[name]:
            raw[name].append(vectors(model, batch))
            updates[name].append(
                virtual_vectors(
                    state,
                    optimizer_state,
                    batch,
                    device,
                    lr=float(manifest["optimizer"]["lr"]),
                    clip=float(manifest["gradient_clip"]),
                )
            )
    raw_cosines = pairwise(raw, seed=191)
    update_cosines = pairwise(updates, seed=192)
    return {
        "row_count": len(rows),
        "game_count": sum(
            len({game_identity(rows[i]) for i in indexes})
            for indexes in assignments.values()
        ),
        "shard_row_counts": {
            name: len(indexes) for name, indexes in assignments.items()
        },
        "raw_gradient_pairwise_cosines": raw_cosines,
        "adam_update_pairwise_cosines": update_cosines,
        "median_joint_trunk_adam_cosine": median_joint_trunk_cosine(update_cosines),
        "median_joint_trunk_raw_gradient_cosine": median_joint_trunk_cosine(
            raw_cosines
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_multi_generation_replay_window"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-multi-generation-replay-window-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-multi-generation-replay-window-results.md",
    )
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, int(manifest["seed"]))

    sources = [(name, load_source(name, Path(path))) for name, path in SOURCES]
    source_summary = {
        name: {
            "path": str(path),
            "row_count": len(rows),
            "sha256": sha256_file(Path(path)),
        }
        for (name, path), (_name, rows) in zip(SOURCES, sources, strict=True)
    }

    state, optimizer_state = fresh_state(manifest, device)

    windows: dict[str, Any] = {}
    previous_rows: list[dict[str, Any]] = []
    for width in range(1, len(SOURCES) + 1):
        previous_rows = [row for name, rows in sources[:width] for row in rows]
        assignments = partition(previous_rows)
        windows[f"W{width}"] = {
            "sources": [name for name, _ in sources[:width]],
            "rows": previous_rows,
            "assignments": assignments,
        }

    # Run progressively, reusing the largest window's batches is not possible
    # because shards differ, so compute each window independently.
    windows_result: dict[str, Any] = {}
    for width in range(1, len(SOURCES) + 1):
        label = f"W{width}"
        entry = windows[label]
        windows_result[label] = window_summary(
            entry["rows"],
            entry["assignments"],
            state,
            optimizer_state,
            device,
            manifest,
        )
        windows_result[label]["sources"] = entry["sources"]

    # Deterministic one-step candidates for the largest window (parity with D/E).
    largest = windows[f"W{len(SOURCES)}"]
    largest_assignments = largest["assignments"]
    first_batches = {
        name: _batch(
            largest["rows"],
            deterministic_batches(indexes, name)[0],
            device,
        )
        for name, indexes in largest_assignments.items()
    }
    candidates = {
        f"joint_{name}_step1": real_candidate(
            state, optimizer_state, first_batches[name], device, manifest
        )
        for name in SHARDS
    }
    repeated = {
        name: real_candidate(
            state, optimizer_state, first_batches[shard], device, manifest
        )
        for name, shard in zip(candidates, SHARDS, strict=True)
    }
    if any(
        state_hash(candidates[name]) != state_hash(repeated[name])
        for name in candidates
    ):
        raise RuntimeError("one-step lane is not deterministic")

    summary: dict[str, Any] = {
        "schema": "azlite_multi_generation_replay_window_v1",
        "guardrails": {
            "new_self_play": False,
            "target_change": False,
            "lr_change": False,
            "loss_weight_change": False,
            "optimizer_steps_per_lane": 1,
            "virtual_steps_chained": False,
            "promotion": False,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "sources": source_summary,
        },
        "windows": windows_result,
        "phase_d": {
            name: {"state_sha256": state_hash(candidate)}
            for name, candidate in candidates.items()
        },
    }
    summary["classification"] = {
        "primary_statistic": "median pairwise joint-trunk Adam-update cosine across shards",
        "baseline_w1": windows_result["W1"]["median_joint_trunk_adam_cosine"],
        "largest_window": windows_result[f"W{len(SOURCES)}"][
            "median_joint_trunk_adam_cosine"
        ],
        "label": classify(windows_result),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "windows": {
                    label: {
                        "rows": entry["row_count"],
                        "median_joint_trunk_adam_cosine": entry[
                            "median_joint_trunk_adam_cosine"
                        ],
                    }
                    for label, entry in windows_result.items()
                },
                "classification": summary["classification"]["label"],
            }
        )
    )
    return 0


def classify(windows_result: dict[str, Any]) -> str:
    largest_key = f"W{len(windows_result)}"
    largest = windows_result[largest_key]["median_joint_trunk_adam_cosine"]
    if largest >= 0.75:
        return "larger_window_stabilizes_update_direction"
    if largest < 0.50:
        return "larger_window_insufficient_direction_still_unstable"
    return "larger_window_partially_stabilizes"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Multi-Generation Replay Window Audit",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        "Primary statistic: median pairwise joint-trunk Adam-update cosine across the four whole-game shards.",
        "",
        "| Window | Sources | Rows | Median joint-trunk Adam cosine |",
        "| --- | --- | ---: | ---: |",
    ]
    for label, entry in summary["windows"].items():
        lines.append(
            f"| {label} | {', '.join(entry['sources'])} | {entry['row_count']} | "
            f"{entry['median_joint_trunk_adam_cosine']:.4f} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: a median joint-trunk Adam-update cosine below 0.50 means the "
        "replay window is still too noisy for stable representation updates; a value at "
        "or above 0.75 means the update direction has stabilized and the supervised "
        "objective/targets themselves should be audited."
    )
    lines.append("")
    lines.append(
        "Full evidence: `docs/data/alphazero-lite-multi-generation-replay-window-summary.json`."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
