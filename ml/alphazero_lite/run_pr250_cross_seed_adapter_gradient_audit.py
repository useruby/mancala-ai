#!/usr/bin/env python3
"""Compare frozen PR #248/#249 adapter deltas and full-batch target gradients.

This is an analysis-only follow-up to fresh-suite generalization.  It loads
existing replays and checkpoints, performs backward passes only, and never
creates self-play, trains, updates an optimizer, or exports a candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (  # noqa: E402
    ADAPTER_KEYS,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _batch,
    _losses,
)
from ml.alphazero_lite.train import apply_trainable_scope, load_checkpoint_into_model  # noqa: E402

CANONICAL_SUITE = Path("/tmp/azlite_opening_suite/medium_eval.jsonl")
A16 = replay.A16_SNAPSHOT
LANES = {
    "seed45_fixed768_positive": {
        "replay": Path("/tmp/azlite_pr247_fixed_target_budget/derived/fixed768.jsonl"),
        "checkpoint": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed768/step_0016.pt"
        ),
        "replay_sha": "aed8a767b9a03a8cfadff617464d2fc218e3bf6006fde72123751e192d48b5a5",
    },
    "seed45_fixed1024_negative": {
        "replay": Path("/tmp/azlite_pr247_fixed_target_budget/derived/fixed1024.jsonl"),
        "checkpoint": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed1024/step_0016.pt"
        ),
        "replay_sha": "7bc74369439aa804548400536ba97bb18ecc2a0c9372320a9c7a3d4fa285092f",
    },
    "seed46_fresh1024_positive": {
        "replay": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/generated/fresh1024.jsonl"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh1024/step_0016.pt"
        ),
        "replay_sha": "007a72d5c07c15f353ef244cab627198cb7be6b73c43b7a03ea2feb767b1fbb4",
    },
    "seed46_fresh768_negative": {
        "replay": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/generated/fresh768.jsonl"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh768/step_0016.pt"
        ),
        "replay_sha": "83afa719a260083902540419b72b30b6a96d7bc63015be2482af442ab7d4baa9",
    },
}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def vector(
    state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor] | None = None
) -> torch.Tensor:
    values = []
    for key in ADAPTER_KEYS:
        value = state[key].detach().cpu()
        values.append(
            (
                value if reference is None else value - reference[key].detach().cpu()
            ).reshape(-1)
        )
    return torch.cat(values)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm, right_norm = (
        torch.linalg.vector_norm(left),
        torch.linalg.vector_norm(right),
    )
    if left_norm == 0 or right_norm == 0:
        fail("zero adapter vector")
    return float(torch.dot(left, right) / (left_norm * right_norm))


def load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.is_file():
        fail(f"missing replay: {path}")
    blocked, groups = replay.exclusion_hashes(CANONICAL_SUITE)
    return replay.filter_rows(read_jsonl(path), blocked, groups)


def full_batch_gradient(
    rows: list[dict[str, Any]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(a16)
    apply_trainable_scope(model, "policy_adapter_only")
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    parent_model.eval()
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    total, weighted_policy_loss, weighted_value_loss = None, 0.0, 0.0
    count = 0
    model.train()
    for indexes in replay.batches(rows):
        batch = _batch(rows, indexes, torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"],
            incumbent_policy_batch(parent_model, batch),
            batch["mask"],
            replay.BETA,
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        model.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        gradient = torch.cat(
            [
                dict(model.named_parameters())[key].grad.detach().cpu().reshape(-1)
                for key in ADAPTER_KEYS
            ]
        )
        batch_size = len(indexes)
        total = (
            gradient * batch_size if total is None else total + (gradient * batch_size)
        )
        weighted_policy_loss += float(policy_loss.detach()) * batch_size
        weighted_value_loss += float(value_loss.detach()) * batch_size
        count += batch_size
    if total is None or count == 0 or not torch.isfinite(total).all():
        fail("invalid full-batch gradient")
    assert total is not None
    gradient = total / count
    return {
        "vector": gradient,
        "norm": float(torch.linalg.vector_norm(gradient)),
        "policy_loss": weighted_policy_loss / count,
        "value_loss": weighted_value_loss / count,
        "rows": count,
        "batches": len(replay.batches(rows)),
    }


def matrix(vectors: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        left: {right: cosine(value, other) for right, other in vectors.items()}
        for left, value in vectors.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr250_cross_seed_adapter_gradient_audit"),
    )
    args = parser.parse_args()
    if not CANONICAL_SUITE.is_file() or not A16.is_file():
        fail("missing canonical suite or A16 snapshot")
    snapshot = torch.load(A16, map_location="cpu", weights_only=False)
    a16, _optimizer = replay.immutable_initial_state(snapshot)
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    deltas, gradients, lanes = {}, {}, {}
    for name, lane in LANES.items():
        if lane["replay_sha"] and sha256_file(lane["replay"]) != lane["replay_sha"]:
            fail(f"replay SHA mismatch: {name}")
        saved = torch.load(lane["checkpoint"], map_location="cpu", weights_only=False)
        rows, exclusions = load_rows(lane["replay"])
        deltas[name] = vector(saved["model"], a16)
        gradients[name] = full_batch_gradient(rows, a16, parent)
        lanes[name] = {
            "checkpoint_sha256": sha256_file(lane["checkpoint"]),
            "replay_sha256": sha256_file(lane["replay"]),
            "eligible_rows": len(rows),
            "exclusions": exclusions,
            "adapter_delta_norm": float(torch.linalg.vector_norm(deltas[name])),
            "full_batch_gradient_norm": gradients[name]["norm"],
            "full_batch_policy_loss": gradients[name]["policy_loss"],
            "full_batch_value_loss": gradients[name]["value_loss"],
        }
    pairs = {
        "positive_deltas_cross_seed": cosine(
            deltas["seed45_fixed768_positive"], deltas["seed46_fresh1024_positive"]
        ),
        "negative_deltas_cross_seed": cosine(
            deltas["seed45_fixed1024_negative"], deltas["seed46_fresh768_negative"]
        ),
        "positive_full_batch_gradients_cross_seed": cosine(
            gradients["seed45_fixed768_positive"]["vector"],
            gradients["seed46_fresh1024_positive"]["vector"],
        ),
        "negative_full_batch_gradients_cross_seed": cosine(
            gradients["seed45_fixed1024_negative"]["vector"],
            gradients["seed46_fresh768_negative"]["vector"],
        ),
    }
    summary = {
        "schema": "alphazero_lite_pr250_cross_seed_adapter_gradient_audit_v1",
        "analysis_only": True,
        "a16_snapshot_sha256": sha256_file(A16),
        "beta": replay.BETA,
        "lanes": lanes,
        "adapter_delta_cosine": matrix(deltas),
        "full_batch_gradient_cosine": matrix(
            {name: value["vector"] for name, value in gradients.items()}
        ),
        "key_cross_seed_cosines": pairs,
        "delta_vs_negative_gradient_cosine": {
            name: cosine(deltas[name], -gradients[name]["vector"]) for name in LANES
        },
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(pairs, sort_keys=True))


if __name__ == "__main__":
    main()
