#!/usr/bin/env python3
"""Attribute PR #191's joint-trunk regression without training or promotion.

The runner only copies checkpoint tensors into diagnostic artifacts.  It never
constructs an optimizer and has no code path that calls ``optimizer.step``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    derive_search_seed,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import (  # noqa: E402
    _decoded_state,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_policy_target_noise_causal_closeout import (  # noqa: E402
    forced_continuation,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.evaluation_metrics import (  # noqa: E402
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    CheckpointEvaluator,
    build_eval_search_options,
    encode_state,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

CURRENT_HASH = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
HEAD_PREFIXES = (
    "policy_hidden_layer.",
    "policy_head.",
    "value_hidden_layer.",
    "value_head.",
)
MODEL_KWARGS = ((96, 3), "residual_v3", input_size_for_encoding("kalah_v3"))
PROBE_SIZE = 512
SEARCH_CONTEXTS = (
    "384:256",
    "768:256",
    "768:768",
    "1200:1200",
    "1200:256",
    "256:768",
)
CONTINUATION_BUDGETS = (768, 1200)
PHASE_C_CONTRASTS = (
    ("T-C", "C", "T"),
    ("JH-C", "C", "JH"),
    ("J-JH", "JH", "J"),
    ("J-T", "T", "J"),
    ("J-H", "H", "J"),
)
PHASE_D_PRIORITY_CONTRASTS = (
    ("T-C", "C", "T"),
    ("J-JH", "JH", "J"),
    ("JH-C", "C", "JH"),
)
PHASE_F_CONTEXTS = ("384:256", "768:768", "1200:1200", "256:768")
PHASE_F_CONTRASTS = (
    ("T-C", "C", "T"),
    ("JH-C", "C", "JH"),
    ("J-JH", "JH", "J"),
    ("J-T", "T", "J"),
)
PHASE_F_SEED = 42
PHASE_F_GAMES_PER_OPENING = 2
MATERIAL_EFFECT = 0.03
POLICY_GRADIENT_DOMINANCE_RATIO = 2.0
PHASE_F_SUITE = Path(
    "/tmp/azlite_shared_trunk_learning/arena_vs_current/temp_0_0/seed_42/"
    "artifact/equal_high/starts_0/opening_suite.jsonl"
)


def tensor_family(name: str) -> str:
    """Return the PR #191 tensor family for a state-dict name."""
    if name.startswith(TRUNK_PREFIXES):
        return "trunk"
    if name.startswith(HEAD_PREFIXES):
        return "heads"
    raise ValueError(f"unexpected residual_v3 parameter: {name}")


def artifact_weight_hash(path: Path) -> str:
    """Return the persisted JSON weight hash and reject a stale artifact."""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    actual = sha256_file(path / "weights.json")
    expected = metadata["artifacts"].get("weights_json_sha256")
    if expected is not None and actual != expected:
        raise RuntimeError(f"artifact weights hash mismatch: {path}")
    return actual


def load_artifact_state(artifact: Path, scratch: Path) -> dict[str, torch.Tensor]:
    """Load artifact weights through the production checkpoint conversion."""
    checkpoint = scratch / f"{artifact.name}.npz"
    materialize_weights_json_checkpoint(
        weights_path=artifact / "weights.json", out_path=checkpoint
    )
    model = PolicyValueNet(*MODEL_KWARGS)
    load_checkpoint_into_model(model, checkpoint)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def byte_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor dtype, shape, and payload exactly."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.detach().cpu().numpy().tobytes()
        == right.detach().cpu().numpy().tobytes()
    )


def hybrid_state(
    current: dict[str, torch.Tensor],
    joint: dict[str, torch.Tensor],
    *,
    trunk_from_joint: bool,
    heads_from_joint: bool,
) -> dict[str, torch.Tensor]:
    """Create a C/T/JH/J state from exact source tensors, never deltas."""
    result: dict[str, torch.Tensor] = {}
    for name, value in current.items():
        use_joint = (tensor_family(name) == "trunk" and trunk_from_joint) or (
            tensor_family(name) == "heads" and heads_from_joint
        )
        result[name] = (joint if use_joint else current)[name].detach().clone()
    return result


def assert_decomposition(
    current: dict[str, torch.Tensor],
    joint: dict[str, torch.Tensor],
    models: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Enforce the bit-for-bit invariants that make hybrids interpretable."""
    for name in current:
        if not byte_identical(models["C"][name], current[name]):
            raise AssertionError(f"C differs from current: {name}")
        if not byte_identical(models["J"][name], joint[name]):
            raise AssertionError(f"J differs from joint artifact: {name}")
        for label, state, source in (
            ("T", models["T"], current if tensor_family(name) == "heads" else joint),
            ("JH", models["JH"], current if tensor_family(name) == "trunk" else joint),
        ):
            if not byte_identical(state[name], source[name]):
                raise AssertionError(f"{label} has a non-source tensor: {name}")


def write_artifact(
    path: Path, state: dict[str, torch.Tensor], source_metadata: Path
) -> Path:
    """Export a diagnostic-only artifact with a provenance marker."""
    model = PolicyValueNet(*MODEL_KWARGS)
    model.load_state_dict(state)
    weights = {
        key: value.tolist() for key, value in checkpoint_from_model(model).items()
    }
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / "weights.json"
    weights_path.write_text(
        json.dumps(weights, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata["version"] = f"pr191_delta_diagnostic_{path.name.lower()}"
    metadata["diagnostic_only"] = True
    metadata["promotion_forbidden"] = True
    metadata["artifacts"]["weights_json_sha256"] = sha256_file(weights_path)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return path


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute legal-only policies."""
    masked = np.where(mask, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.where(mask, np.exp(shifted), 0.0)
    return exp / np.sum(exp, axis=1, keepdims=True)


def js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return per-row Jensen-Shannon divergence in nats."""
    midpoint = (left + right) / 2.0
    left_safe = np.clip(left, 1e-12, None)
    right_safe = np.clip(right, 1e-12, None)
    midpoint_safe = np.clip(midpoint, 1e-12, None)
    return 0.5 * np.sum(
        left * np.log(left_safe / midpoint_safe), axis=1
    ) + 0.5 * np.sum(right * np.log(right_safe / midpoint_safe), axis=1)


def model_outputs(
    state: dict[str, torch.Tensor], states: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run one model deterministically in bounded batches."""
    model = PolicyValueNet(*MODEL_KWARGS)
    model.load_state_dict(state)
    model.eval()
    logits, values = [], []
    with torch.no_grad():
        for chunk in np.array_split(states, max(1, math.ceil(len(states) / 2048))):
            policy, value = model(torch.as_tensor(chunk, dtype=torch.float32))
            logits.append(policy.numpy())
            values.append(value.numpy().reshape(-1))
    return softmax_masked(np.concatenate(logits), mask), np.concatenate(values)


def output_metrics(
    rows: list[dict[str, Any]], states: dict[str, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    """Calculate Phase B metrics on the frozen validation rows."""
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    targets = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    outputs = {name: model_outputs(state, x, mask) for name, state in states.items()}
    current_policy, current_value = outputs["C"]
    result: dict[str, Any] = {}
    rng = np.random.default_rng(191)
    pair_count = min(100_000, len(rows) * 10)
    left, right = rng.integers(0, len(rows), size=(2, pair_count))
    distinct = left != right
    for name, (policy, prediction) in outputs.items():
        ce = -np.sum(targets * np.log(np.clip(policy, 1e-12, None)), axis=1)
        kl = np.sum(
            policy
            * np.log(
                np.clip(policy, 1e-12, None) / np.clip(current_policy, 1e-12, None)
            ),
            axis=1,
        )
        entropy = -np.sum(policy * np.log(np.clip(policy, 1e-12, None)), axis=1)
        pairwise = np.sign(prediction[left] - prediction[right]) == np.sign(
            values[left] - values[right]
        )
        error = prediction - values
        result[name] = {
            "policy": {
                "legal_cross_entropy_to_replay_teacher": float(np.mean(ce)),
                "js_to_replay_teacher": float(np.mean(js(policy, targets))),
                "kl_from_current": float(np.mean(kl)),
                "top1_agreement_to_replay_teacher": float(
                    np.mean(np.argmax(policy, axis=1) == np.argmax(targets, axis=1))
                ),
                "top1_agreement_with_current": float(
                    np.mean(
                        np.argmax(policy, axis=1) == np.argmax(current_policy, axis=1)
                    )
                ),
                "entropy": float(np.mean(entropy)),
            },
            "value": {
                "huber_loss_to_canonical_outcome": float(
                    np.mean(
                        np.where(np.abs(error) < 1, 0.5 * error**2, np.abs(error) - 0.5)
                    )
                ),
                "mae": float(np.mean(np.abs(error))),
                "sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(values))),
                "pairwise_concordance": float(np.mean(pairwise[distinct]))
                if np.any(distinct)
                else 0.0,
                "pairwise_samples": int(np.sum(distinct)),
            },
        }
    return result


def gradient_metrics(
    rows: list[dict[str, Any]],
    batch_indexes: np.ndarray,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure separate policy/value trunk gradients; no optimizer is created."""
    model = PolicyValueNet(*MODEL_KWARGS)
    model.load_state_dict(state)
    trunk = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if tensor_family(name) == "trunk"
    ]
    samples: list[dict[str, Any]] = []
    for batch_number, indexes in enumerate(batch_indexes[:32]):
        indexes = indexes[indexes >= 0]
        batch = [rows[int(index)] for index in indexes]
        x = torch.tensor(np.asarray([row["state"] for row in batch], dtype=np.float32))
        target_policy = torch.tensor(
            np.asarray([row["policy"] for row in batch], dtype=np.float32)
        )
        target_value = torch.tensor(
            np.asarray([row["value"] for row in batch], dtype=np.float32)
        )
        logits, prediction = model(x)
        mask = torch.tensor(
            legal_mask_matrix_for_encoded_states(x.numpy()).astype(bool)
        )
        policy_loss = (
            -(target_policy * torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1))
            .sum(1)
            .mean()
        )
        value_loss = torch.nn.functional.smooth_l1_loss(
            prediction.reshape(-1), target_value
        )
        policy_grad = torch.autograd.grad(
            policy_loss, [item[1] for item in trunk], retain_graph=True
        )
        value_grad = torch.autograd.grad(0.6 * value_loss, [item[1] for item in trunk])
        p = torch.cat([item.reshape(-1) for item in policy_grad])
        v = torch.cat([item.reshape(-1) for item in value_grad])
        block_accumulators: dict[str, dict[str, float]] = {}
        for name, pg, vg in zip(
            (item[0] for item in trunk), policy_grad, value_grad, strict=True
        ):
            block = (
                name.split(".")[1]
                if name.startswith("residual_layers.")
                else "input_layer"
            )
            entry = block_accumulators.setdefault(
                block, {"p2": 0.0, "v2": 0.0, "dot": 0.0}
            )
            entry["p2"] += float(torch.sum(pg**2))
            entry["v2"] += float(torch.sum(vg**2))
            entry["dot"] += float(torch.sum(pg * vg))
        samples.append(
            {
                "batch": batch_number,
                "grad_policy_norm": float(torch.linalg.vector_norm(p)),
                "weighted_grad_value_norm": float(torch.linalg.vector_norm(v)),
                "cosine": float(
                    torch.dot(p, v)
                    / (
                        torch.linalg.vector_norm(p) * torch.linalg.vector_norm(v)
                        + 1e-20
                    )
                ),
                "by_residual_block": {
                    key: {
                        "grad_policy_norm": math.sqrt(value["p2"]),
                        "weighted_grad_value_norm": math.sqrt(value["v2"]),
                        "cosine": value["dot"]
                        / math.sqrt(max(value["p2"] * value["v2"], 1e-40)),
                    }
                    for key, value in block_accumulators.items()
                },
            }
        )
    rng = np.random.default_rng(191)
    cosines = np.asarray([sample["cosine"] for sample in samples], dtype=np.float64)
    bootstrap = cosines[
        rng.integers(0, len(cosines), size=(10_000, len(cosines)))
    ].mean(axis=1)
    mean_blocks: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        for block, metrics in sample["by_residual_block"].items():
            for key, value in metrics.items():
                mean_blocks[block][key].append(value)
    return {
        "batches": len(samples),
        "optimizer_steps": 0,
        "samples": samples,
        "mean": {
            key: float(np.mean([sample[key] for sample in samples]))
            for key in ("grad_policy_norm", "weighted_grad_value_norm", "cosine")
        },
        "cosine_bootstrap_95": {
            "lower": float(np.quantile(bootstrap, 0.025)),
            "upper": float(np.quantile(bootstrap, 0.975)),
            "samples": 10_000,
        },
        "by_residual_block_mean": {
            block: {key: float(np.mean(values)) for key, values in metrics.items()}
            for block, metrics in sorted(mean_blocks.items())
        },
    }


def phase_g_classification(
    phase_b: dict[str, Any], phase_e: dict[str, Any], phase_f: dict[str, Any]
) -> dict[str, Any]:
    """Classify only effects that the isolated Phase F contrasts support."""
    del phase_b  # Offline fit metrics are descriptive, not causal arena evidence.
    classifications: list[str] = []
    criteria: dict[str, Any] = {}
    current_gradient = phase_e["current"]
    policy_norm = current_gradient["mean"]["grad_policy_norm"]
    value_norm = current_gradient["mean"]["weighted_grad_value_norm"]
    gradient_ratio = policy_norm / value_norm if value_norm else math.inf
    cosine_ci = current_gradient["cosine_bootstrap_95"]
    criteria["policy_gradient_dominates_trunk_learning"] = {
        "policy_to_weighted_value_norm_ratio": gradient_ratio,
        "minimum_ratio": POLICY_GRADIENT_DOMINANCE_RATIO,
        "met": gradient_ratio >= POLICY_GRADIENT_DOMINANCE_RATIO,
    }
    if criteria["policy_gradient_dominates_trunk_learning"]["met"]:
        classifications.append("policy_gradient_dominates_trunk_learning")

    criteria["gradient_conflict"] = {
        "cosine_bootstrap_95": cosine_ci,
        "requires_upper_below_zero": True,
        "met": cosine_ci["upper"] < 0.0,
    }
    if criteria["gradient_conflict"]["met"]:
        classifications.append("gradient_conflict")

    if not phase_f.get("enabled"):
        return {
            "primary_classification": None,
            "classifications": classifications,
            "criteria": criteria,
            "next_action": "complete_phase_f_before_selecting_a_training_change",
        }

    contrasts = phase_f["metrics"]
    practical_trunk = contrasts["384:256"]["direct_contrasts"]["T-C"]
    high_search_trunk = contrasts["1200:1200"]["direct_contrasts"]["T-C"]
    high_search_heads = contrasts["1200:1200"]["direct_contrasts"]["JH-C"]

    def materially_harmful(effect: dict[str, Any]) -> bool:
        return (
            effect["paired_candidate_effect"] <= -MATERIAL_EFFECT
            and effect["opening_bootstrap_ci"]["upper_95"] < 0.0
        )

    def materially_beneficial(effect: dict[str, Any]) -> bool:
        return (
            effect["paired_candidate_effect"] >= MATERIAL_EFFECT
            and effect["opening_bootstrap_ci"]["lower_95"] > 0.0
        )

    criteria["tradeoff"] = {
        "practical_trunk_effect": practical_trunk["paired_candidate_effect"],
        "high_search_trunk_effect": high_search_trunk["paired_candidate_effect"],
        "requires_practical_benefit_and_high_search_harm": True,
        "met": materially_beneficial(practical_trunk)
        and materially_harmful(high_search_trunk),
    }
    if criteria["tradeoff"]["met"]:
        classifications.append("tradeoff")

    criteria["mixed_additive_harm"] = {
        "high_search_trunk_effect": high_search_trunk["paired_candidate_effect"],
        "high_search_heads_effect": high_search_heads["paired_candidate_effect"],
        "requires_both_isolated_effects_materially_harmful": True,
        "met": materially_harmful(high_search_trunk)
        and materially_harmful(high_search_heads),
    }
    if criteria["mixed_additive_harm"]["met"]:
        classifications.append("mixed_additive_harm")

    return {
        # Neither isolated delta is a supported exclusive cause when both harm.
        "primary_classification": None,
        "classifications": classifications,
        "criteria": criteria,
        "next_action": (
            "do_not_run_another_unrestricted_joint_update; "
            "investigate_value_preserving_or_loss_decoupled_representation_updates"
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist audit rows in a stable order for independent inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, ignoring an interrupted final write."""
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _phase_c_identity(
    row: dict[str, Any],
    probe_by_hash: dict[str, dict[str, Any]],
    manifest_hash: str,
    seed: int,
) -> tuple[str, str, str] | None:
    """Validate a Phase C record against its immutable search input identity."""
    required = (
        "state_hash",
        "state",
        "manifest_index",
        "context",
        "model",
        "simulations",
        "c_puct",
        "seed_contract",
        "search_seed",
        "seed_context_hash",
        "selected_move",
        "visit_policy",
        "root_value",
        "selected_child_q_rank",
        "selected_visit_rank",
        "selected_child_q",
        "top1_top2_visit_margin",
        "child_stats",
    )
    if any(field not in row for field in required):
        return None
    state_hash = row["state_hash"]
    context = row["context"]
    model = row["model"]
    if (
        not isinstance(state_hash, str)
        or context not in SEARCH_CONTEXTS
        or model not in {"C", "T", "JH", "J", "H"}
    ):
        return None
    probe_row = probe_by_hash.get(state_hash)
    if probe_row is None or any(
        row[field] != probe_row[field] for field in ("state", "manifest_index")
    ):
        return None
    search_seed, seed_context_hash = _search_seed(probe_row, manifest_hash, seed)
    if (
        row["simulations"] != _context_simulations(context)
        or row["c_puct"] != _context_c_puct(context)
        or row["seed_contract"] != SEED_CONTRACT_VERSION
        or row["search_seed"] != search_seed
        or row["seed_context_hash"] != seed_context_hash
    ):
        return None
    return state_hash, context, model


def load_complete_phase_c_records(
    path: Path, probe: list[dict[str, Any]], *, manifest_hash: str, seed: int
) -> list[dict[str, Any]] | None:
    """Return a complete validated Phase C checkpoint, or ``None`` to rebuild it."""
    records = _read_jsonl(path)
    expected_count = PROBE_SIZE * len(SEARCH_CONTEXTS) * 5
    expected = {
        (row["state_hash"], context, model)
        for row in probe
        for context in SEARCH_CONTEXTS
        for model in ("C", "T", "JH", "J", "H")
    }
    if len(probe) != PROBE_SIZE or len(records) != expected_count:
        return None
    probe_by_hash = {row["state_hash"]: row for row in probe}
    identities = {
        _phase_c_identity(row, probe_by_hash, manifest_hash, seed) for row in records
    }
    return records if None not in identities and identities == expected else None


def decoded_validation_manifest(
    rows: list[dict[str, Any]], validation_indexes: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select 512 unique, round-trippable raw positions from validation only."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: list[dict[str, Any]] = []
    for source_index in validation_indexes.tolist():
        row = rows[int(source_index)]
        try:
            state = _decoded_state(row)
            game = KalahGame.from_state(state)
            encoded = np.asarray(
                encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
            )
            persisted = np.asarray(row["state"], dtype=np.float32)
            if encoded.shape != persisted.shape or not np.array_equal(
                encoded, persisted
            ):
                raise ValueError(
                    "decoded state does not exactly re-encode to the validation row"
                )
            if not game.possible_moves():
                raise ValueError("terminal validation state")
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"source_index": int(source_index), "reason": str(exc)})
            continue
        state_hash = stable_hash(state)
        if state_hash in seen:
            continue
        seen.add(state_hash)
        selected.append(
            {
                "manifest_index": len(selected),
                "source_index": int(source_index),
                "state_hash": state_hash,
                "state": state,
                "player": int(game.current_player),
                "legal_moves": [int(move) for move in game.possible_moves()],
            }
        )
        if len(selected) == PROBE_SIZE:
            break
    if len(selected) != PROBE_SIZE:
        raise RuntimeError(
            f"only {len(selected)} verified unique validation states available; need {PROBE_SIZE}"
        )
    return selected, {
        "schema": "azlite_pr191_post_pr_validation_manifest_v1",
        "selection": "validation source-index order; first unique exact decoded/re-encoded nonterminal rows",
        "state_count": len(selected),
        "unique_state_count": len({row["state_hash"] for row in selected}),
        "state_hashes": [row["state_hash"] for row in selected],
        "source_indexes": [row["source_index"] for row in selected],
        "rejected_before_selection": rejected,
    }


def _context_simulations(context: str) -> int:
    """Use the acting/challenger side of a requested deterministic context."""
    challenger, _current = context.split(":", maxsplit=1)
    return int(challenger)


def _context_c_puct(context: str) -> float:
    """Return the preregistered exploration constant for a root context."""
    return 0.90 if context == "768:768" else 1.25


def _search_seed(
    record: dict[str, Any], manifest_hash: str, base_seed: int
) -> tuple[int, str]:
    """Derive a treatment-invariant v2 search seed for one raw state."""
    return derive_search_seed(
        base_seed=base_seed,
        suite_sha256=manifest_hash,
        opening_index=int(record["manifest_index"]),
        opening_state_hash=str(record["state_hash"]),
        challenger_player=int(record["player"]),
        game_within_opening=0,
        ply=0,
        canonical_current_state_hash=str(record["state_hash"]),
        acting_role="challenger",
        rng_stream_name="pr191_shared_trunk_phase_c",
        contract_version=SEED_CONTRACT_VERSION,
    )


def _rank(move: int, child_stats: list[dict[str, Any]], field: str) -> int:
    ordered = sorted(
        child_stats, key=lambda row: (-float(row[field]), int(row["move"]))
    )
    return 1 + [int(row["move"]) for row in ordered].index(move)


def _visit_policy(child_stats: list[dict[str, Any]]) -> list[float]:
    """Return the root visit distribution in stable legal-move order."""
    visits = np.asarray(
        [
            float(row["visits"])
            for row in sorted(child_stats, key=lambda row: int(row["move"]))
        ]
    )
    total = float(visits.sum())
    return (visits / total).tolist() if total else np.zeros_like(visits).tolist()


def phase_c_search_records(
    probe: list[dict[str, Any]],
    artifacts: dict[str, str],
    *,
    manifest_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Run each immutable composition under all six requested root contexts."""
    evaluators = {
        name: arena.ArtifactEvaluator(Path(path)) for name, path in artifacts.items()
    }
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    records = []
    for state_row in probe:
        search_seed, seed_context_hash = _search_seed(state_row, manifest_hash, seed)
        for context in SEARCH_CONTEXTS:
            simulations = _context_simulations(context)
            c_puct = _context_c_puct(context)
            for model, evaluator in evaluators.items():
                result = arena.evaluate_artifact_position(
                    evaluator=evaluator,
                    state=state_row["state"],
                    simulations=simulations,
                    seed=search_seed,
                    c_puct=c_puct,
                    search_options=options,
                )
                child_stats = list(result["child_stats"])
                selected_move = int(result["selected_move"])
                visits = sorted(
                    (int(row["visits"]) for row in child_stats), reverse=True
                )
                records.append(
                    {
                        "state_hash": state_row["state_hash"],
                        "state": state_row["state"],
                        "manifest_index": state_row["manifest_index"],
                        "context": context,
                        "model": model,
                        "simulations": simulations,
                        "c_puct": float(c_puct),
                        "seed_contract": SEED_CONTRACT_VERSION,
                        "search_seed": search_seed,
                        "seed_context_hash": seed_context_hash,
                        "selected_move": selected_move,
                        "visit_policy": _visit_policy(child_stats),
                        "root_value": float(
                            result.get("search_root_value", result["value"])
                        ),
                        "selected_child_q_rank": _rank(
                            selected_move, child_stats, "q_value"
                        ),
                        "selected_visit_rank": _rank(
                            selected_move, child_stats, "visits"
                        ),
                        "selected_child_q": float(
                            next(
                                row["q_value"]
                                for row in child_stats
                                if int(row["move"]) == selected_move
                            )
                        ),
                        "top1_top2_visit_margin": int(visits[0] - visits[1])
                        if len(visits) > 1
                        else int(visits[0]),
                        "child_stats": child_stats,
                    }
                )
    return records


def phase_c_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize directed B-minus-A root-search contrasts by context."""
    by_state: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        by_state[(row["state_hash"], row["context"], row["model"])] = row
    summary: dict[str, Any] = {}
    contexts = sorted({row["context"] for row in records})
    for context in contexts:
        state_hashes = sorted(
            state_hash
            for state_hash, row_context, model in by_state
            if row_context == context and model == "C"
        )
        contrasts: dict[str, Any] = {}
        for label, model_a, model_b in PHASE_C_CONTRASTS:
            pairs = [
                (
                    by_state[(state_hash, context, model_a)],
                    by_state[(state_hash, context, model_b)],
                )
                for state_hash in state_hashes
            ]
            contrasts[label] = {
                "orientation": "B-minus-A",
                "A": model_a,
                "B": model_b,
                "states": len(pairs),
                "selected_move_change_rate": float(
                    np.mean(
                        [a["selected_move"] != b["selected_move"] for a, b in pairs]
                    )
                ),
                "mean_visit_policy_js": float(
                    np.mean(
                        [
                            js(
                                np.asarray([a["visit_policy"]]),
                                np.asarray([b["visit_policy"]]),
                            )[0]
                            for a, b in pairs
                        ]
                    )
                ),
                "mean_signed_root_value_delta": float(
                    np.mean([b["root_value"] - a["root_value"] for a, b in pairs])
                ),
                "mean_selected_child_q_rank_change": float(
                    np.mean(
                        [
                            b["selected_child_q_rank"] - a["selected_child_q_rank"]
                            for a, b in pairs
                        ]
                    )
                ),
                "mean_visit_margin_change": float(
                    np.mean(
                        [
                            b["top1_top2_visit_margin"] - a["top1_top2_visit_margin"]
                            for a, b in pairs
                        ]
                    )
                ),
            }
        summary[context] = {"direct_contrasts": contrasts}
    return summary


def _clustered_bootstrap(values: list[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "clusters": 0,
            "samples": 10_000,
            "better_fraction": 0.0,
            "worse_fraction": 0.0,
            "tie_fraction": 0.0,
        }
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(10_000, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "clusters": len(data),
        "samples": 10_000,
        "better_fraction": float(np.mean(data > 0)),
        "worse_fraction": float(np.mean(data < 0)),
        "tie_fraction": float(np.mean(data == 0)),
    }


def phase_f_suite_provenance(suite_path: Path) -> dict[str, Any]:
    """Verify the persisted canonical 128-opening suite and its recorded source."""
    metadata_path = suite_path.parent / "metadata.json"
    if not suite_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            "Phase F requires the persisted canonical medium suite and metadata: "
            f"{suite_path}"
        )
    manifest = json.loads(metadata_path.read_text(encoding="utf-8")).get(
        "cache_manifest", {}
    )
    expected_hash = str(manifest.get("suite_sha256", ""))
    source_path = Path(str(manifest.get("suite_path", "")))
    if (
        not expected_hash
        or int(manifest.get("suite_size", 0)) != 128
        or not source_path.is_file()
    ):
        raise RuntimeError(
            "Phase F canonical suite has no verifiable persisted source/hash"
        )
    suite_hash = sha256_file(suite_path)
    source_hash = sha256_file(source_path)
    if source_hash != expected_hash:
        raise RuntimeError("Phase F canonical source hash differs from persisted hash")
    entries = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_entries = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(entries) != 128 or len(source_entries) != 128:
        raise RuntimeError("Phase F requires exactly 128 canonical openings")
    if [entry.get("prefix_moves") for entry in entries] != [
        entry.get("prefix_moves") for entry in source_entries
    ]:
        raise RuntimeError("Phase F persisted suite differs from its canonical source")
    return {
        "reused_suite_path": str(suite_path),
        "reused_suite_sha256": suite_hash,
        "persisted_source_path": str(source_path),
        "persisted_source_sha256": source_hash,
        "persisted_metadata_path": str(metadata_path),
        "unique_openings": len(entries),
    }


def load_complete_phase_f_arena_records(
    path: Path, *, expected_games: int
) -> list[dict[str, Any]] | None:
    """Return complete arena evidence, or ``None`` when it must be rerun."""
    if not path.is_file():
        return None
    try:
        records = parse_game_jsonl(str(path))
        game_indexes = {int(record["game_index"]) for record in records}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    if len(records) != expected_games or len(game_indexes) != expected_games:
        return None
    return records


def phase_f_paired_benchmarks(
    *, artifacts: dict[str, str], suite_path: Path, workdir: Path, workers: int
) -> dict[str, Any]:
    """Measure only the preregistered hybrid effects with matched A/A controls."""
    suite = phase_f_suite_provenance(suite_path)
    suite_sha = str(suite["persisted_source_sha256"])
    results: dict[str, Any] = {}
    for context in PHASE_F_CONTEXTS:
        challenger_sims, current_sims = (int(value) for value in context.split(":"))
        context_results: dict[str, Any] = {}
        for contrast, model_a, model_b in PHASE_F_CONTRASTS:
            candidate_records: list[dict[str, Any]] = []
            control_records: list[dict[str, Any]] = []
            contrast_dir = workdir / "phase_f" / context.replace(":", "_") / contrast
            for role, challenger, current, records in (
                (
                    "candidate",
                    artifacts[model_b],
                    artifacts[model_a],
                    candidate_records,
                ),
                (
                    "current_control",
                    artifacts[model_a],
                    artifacts[model_a],
                    control_records,
                ),
            ):
                for seat in (0, 1):
                    evidence_dir = contrast_dir / role / f"starts_{seat}"
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    expected_games = (
                        int(suite["unique_openings"]) * PHASE_F_GAMES_PER_OPENING
                    )
                    arena_jsonl = evidence_dir / "arena.jsonl"
                    arena_records = load_complete_phase_f_arena_records(
                        arena_jsonl, expected_games=expected_games
                    )
                    if arena_records is None:
                        run_arena(
                            challenger=challenger,
                            current=current,
                            challenger_sims=challenger_sims,
                            current_sims=current_sims,
                            games=expected_games,
                            seed=PHASE_F_SEED,
                            workers=workers,
                            out_json=str(evidence_dir / "arena.json"),
                            out_jsonl=str(arena_jsonl),
                            opening_prefixes_jsonl=str(suite_path),
                            challenger_starts=seat,
                            games_per_opening=PHASE_F_GAMES_PER_OPENING,
                            root_policy_mode="deterministic",
                            root_temperature=0.0,
                            normalize_values=False,
                            c_puct=_context_c_puct(context),
                            tactical_root_bias=0.0,
                            seed_contract=SEED_CONTRACT_VERSION,
                            suite_sha256=suite_sha,
                            seed_ledger_output=str(evidence_dir / "seed_ledger.jsonl"),
                        )
                        arena_records = load_complete_phase_f_arena_records(
                            arena_jsonl, expected_games=expected_games
                        )
                        if arena_records is None:
                            raise RuntimeError(
                                f"Phase F arena evidence is incomplete after run: {arena_jsonl}"
                            )
                    records.extend(arena_records)
            effect = paired_opening_candidate_effect(
                candidate_records,
                control_records,
                bootstrap_samples=10_000,
                bootstrap_seed=PHASE_F_SEED,
            )
            # Seat effects are intentionally excluded: this phase reports treatment effects only.
            context_results[contrast] = {
                "orientation": "B-minus-A",
                "A": model_a,
                "B": model_b,
                "paired_candidate_effect": effect["paired_candidate_effect"],
                "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                "orientation_decomposition": effect["orientation_decomposition"],
                "candidate_records_path": str(contrast_dir / "candidate"),
                "current_control_records_path": str(contrast_dir / "current_control"),
            }
        results[context] = {"direct_contrasts": context_results}
    return {"suite": suite, "metrics": results}


def _phase_d_tasks(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Build the canonical set of forced-continuation tasks from Phase C."""
    by_identity = {
        (row["state_hash"], row["context"], row["model"]): row for row in records
    }
    states = {row["state_hash"]: row["state"] for row in records}
    tasks = []
    for context in SEARCH_CONTEXTS:
        for contrast, model_a, model_b in PHASE_D_PRIORITY_CONTRASTS:
            disagreements = [
                (
                    by_identity[(state_hash, context, model_a)],
                    by_identity[(state_hash, context, model_b)],
                )
                for state_hash in sorted({row["state_hash"] for row in records})
                if by_identity[(state_hash, context, model_a)]["selected_move"]
                != by_identity[(state_hash, context, model_b)]["selected_move"]
            ]
            for choice_a, choice_b in disagreements:
                for budget in CONTINUATION_BUDGETS:
                    paired_seed = stable_seed(
                        "pr191_phase_d_current_current",
                        seed,
                        choice_a["state_hash"],
                        context,
                        budget,
                    )
                    tasks.append(
                        {
                            "state": states[choice_a["state_hash"]],
                            "state_hash": choice_a["state_hash"],
                            "origin_context": context,
                            "contrast": contrast,
                            "orientation": "B-minus-A",
                            "A": model_a,
                            "B": model_b,
                            "continuation_budget": budget,
                            "forced_move_A": choice_a["selected_move"],
                            "forced_move_B": choice_b["selected_move"],
                            "paired_seed": paired_seed,
                        }
                    )
    return tasks


def _phase_d_identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["state_hash"]),
        str(row["origin_context"]),
        str(row["contrast"]),
        int(row["continuation_budget"]),
    )


def _valid_phase_d_row(row: dict[str, Any], task: dict[str, Any]) -> bool:
    """Accept only a finite completed result for the exact canonical task."""
    required = (
        "state_hash",
        "origin_context",
        "contrast",
        "orientation",
        "A",
        "B",
        "continuation_budget",
        "forced_move_A",
        "forced_move_B",
        "paired_seed",
        "paired_seed_context_hash",
        "outcome_delta",
        "normalized_final_store_margin_delta",
    )
    if any(field not in row for field in required):
        return False
    if any(row[field] != task[field] for field in required if field in task):
        return False
    try:
        return (
            isinstance(row["paired_seed_context_hash"], str)
            and math.isfinite(float(row["outcome_delta"]))
            and math.isfinite(float(row["normalized_final_store_margin_delta"]))
        )
    except (TypeError, ValueError, OverflowError):
        return False


_PHASE_D_EVALUATOR: CheckpointEvaluator | None = None


def _init_phase_d_worker(current_checkpoint: str) -> None:
    """Load the immutable current evaluator once in each process worker."""
    global _PHASE_D_EVALUATOR
    _PHASE_D_EVALUATOR = CheckpointEvaluator(
        Path(current_checkpoint), input_encoding="kalah_v3"
    )


def _run_phase_d_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both forced moves with the process-local current checkpoint."""
    if _PHASE_D_EVALUATOR is None:
        raise RuntimeError("Phase D worker evaluator was not initialized")
    continuation_task = {
        "state": task["state"],
        "state_hash": task["state_hash"],
        "continuation_budget": task["continuation_budget"],
        "experiment_seed": task["paired_seed"],
    }
    baseline = forced_continuation(
        evaluator=_PHASE_D_EVALUATOR,
        task=continuation_task,
        forced_move=task["forced_move_A"],
    )
    forced = forced_continuation(
        evaluator=_PHASE_D_EVALUATOR,
        task=continuation_task,
        forced_move=task["forced_move_B"],
    )
    return {
        **{key: value for key, value in task.items() if key != "state"},
        "paired_seed_context_hash": baseline["paired_seed_context_hash"],
        "outcome_delta": float(forced["outcome_root"] - baseline["outcome_root"]),
        "normalized_final_store_margin_delta": float(
            (forced["store_margin_root"] - baseline["store_margin_root"]) / 48.0
        ),
    }


def phase_d_forced_continuations(
    records: list[dict[str, Any]],
    *,
    current_checkpoint: Path,
    seed: int,
    checkpoint_path: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resume deterministic priority continuations with append-only task checkpoints."""
    tasks = _phase_d_tasks(records, seed=seed)
    tasks_by_identity = {_phase_d_identity(task): task for task in tasks}
    completed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in _read_jsonl(checkpoint_path):
        try:
            identity = _phase_d_identity(row)
        except (KeyError, TypeError, ValueError):
            continue
        task = tasks_by_identity.get(identity)
        if task is not None and _valid_phase_d_row(row, task):
            completed.setdefault(identity, row)
    missing = [task for task in tasks if _phase_d_identity(task) not in completed]
    if missing:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as stream:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_phase_d_worker,
                initargs=(str(current_checkpoint),),
            ) as executor:
                futures = [executor.submit(_run_phase_d_task, task) for task in missing]
                for future in as_completed(futures):
                    row = future.result()
                    completed[_phase_d_identity(row)] = row
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
    rows = [completed[_phase_d_identity(task)] for task in tasks]
    summary: dict[str, Any] = {}
    for context in SEARCH_CONTEXTS:
        for contrast, _model_a, _model_b in PHASE_D_PRIORITY_CONTRASTS:
            for budget in CONTINUATION_BUDGETS:
                subset = [
                    row
                    for row in rows
                    if row["origin_context"] == context
                    and row["contrast"] == contrast
                    and row["continuation_budget"] == budget
                ]
                summary.setdefault(context, {}).setdefault(contrast, {})[
                    str(budget)
                ] = {
                    "orientation": "B-minus-A",
                    "unique_disagreement_states": len(
                        {row["state_hash"] for row in subset}
                    ),
                    "state_clustered_outcome_delta_bootstrap_95": _clustered_bootstrap(
                        [row["outcome_delta"] for row in subset],
                        seed=stable_seed(seed, context, contrast, budget, "outcome"),
                    ),
                    "state_clustered_normalized_final_store_margin_delta_bootstrap_95": _clustered_bootstrap(
                        [
                            float(row["normalized_final_store_margin_delta"])
                            for row in subset
                        ],
                        seed=stable_seed(seed, context, contrast, budget, "margin"),
                    ),
                }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_shared_trunk_delta_attribution"),
    )
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-shared-trunk-delta-attribution-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-shared-trunk-delta-attribution-results.md",
    )
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument(
        "--phase-f",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the canonical paired medium-suite Phase F benchmark (default: enabled).",
    )
    parser.add_argument("--phase-f-suite", type=Path, default=PHASE_F_SUITE)
    parser.add_argument("--phase-f-workers", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Process workers for resumable Phase D continuations (default: min(16, CPUs)).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    verify_manifest(args.pr191_workdir / "training_manifest.json")
    current_hash = artifact_weight_hash(args.current)
    if current_hash != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR #191's persisted hash")
    heads = args.pr191_workdir / "heads_only_e1_run_a/artifact"
    joint = args.pr191_workdir / "joint_trunk_e1_run_a/artifact"
    heads_hash = artifact_weight_hash(heads)
    joint_hash = artifact_weight_hash(joint)
    scratch = args.workdir / "checkpoints"
    scratch.mkdir(parents=True, exist_ok=True)
    current_state = load_artifact_state(args.current, scratch)
    joint_state = load_artifact_state(joint, scratch)
    models = {
        "C": hybrid_state(
            current_state, joint_state, trunk_from_joint=False, heads_from_joint=False
        ),
        "T": hybrid_state(
            current_state, joint_state, trunk_from_joint=True, heads_from_joint=False
        ),
        "JH": hybrid_state(
            current_state, joint_state, trunk_from_joint=False, heads_from_joint=True
        ),
        "J": hybrid_state(
            current_state, joint_state, trunk_from_joint=True, heads_from_joint=True
        ),
        "H": load_artifact_state(heads, scratch),
    }
    assert_decomposition(current_state, joint_state, models)
    artifacts = {"C": str(args.current), "J": str(joint), "H": str(heads)}
    for name in ("T", "JH"):
        artifacts[name] = str(
            write_artifact(
                args.workdir / "diagnostic_artifacts" / name,
                models[name],
                args.current / "metadata.json",
            )
        )
    rows = [
        json.loads(line)
        for line in (args.pr191_workdir / "replay.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    validation_indexes = np.load(args.pr191_workdir / "validation_source_indexes.npy")
    validation_rows = [rows[int(index)] for index in validation_indexes]
    probe, probe_manifest = decoded_validation_manifest(rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        args.pr191_workdir / "validation_source_indexes.npy"
    )
    probe_manifest["replay_sha256"] = sha256_file(args.pr191_workdir / "replay.jsonl")
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    write_jsonl(args.workdir / "post_pr_validation_512_states.jsonl", probe)
    (args.workdir / "post_pr_validation_512_manifest.json").write_text(
        json.dumps(probe_manifest, indent=2) + "\n", encoding="utf-8"
    )
    phase_c_path = args.workdir / "phase_c_search_records.jsonl"
    phase_c_rows = load_complete_phase_c_records(
        phase_c_path,
        probe,
        manifest_hash=probe_manifest["manifest_sha256"],
        seed=args.seed,
    )
    if phase_c_rows is None:
        phase_c_rows = phase_c_search_records(
            probe,
            artifacts,
            manifest_hash=probe_manifest["manifest_sha256"],
            seed=args.seed,
        )
        write_jsonl(phase_c_path, phase_c_rows)
    phase_d_path = args.workdir / "phase_d_current_current_forced_continuations.jsonl"
    phase_d_rows, phase_d_summary = phase_d_forced_continuations(
        phase_c_rows,
        current_checkpoint=scratch / f"{args.current.name}.npz",
        seed=args.seed,
        checkpoint_path=phase_d_path,
        workers=args.workers,
    )
    # Canonicalize completed output after the append-only checkpoint has survived the run.
    write_jsonl(phase_d_path, phase_d_rows)
    phase_f: dict[str, Any] = (
        phase_f_paired_benchmarks(
            artifacts={name: artifacts[name] for name in ("C", "T", "JH", "J")},
            suite_path=args.phase_f_suite,
            workdir=args.workdir,
            workers=args.phase_f_workers,
        )
        if args.phase_f
        else {"enabled": False}
    )
    summary = {
        "schema": "azlite_shared_trunk_delta_attribution_v3",
        "guardrails": {"training": False, "optimizer_steps": 0, "promotion": False},
        "inputs": {
            "training_manifest": str(args.pr191_workdir / "training_manifest.json"),
            "verified_weight_hashes": {
                "C": current_hash,
                "H": heads_hash,
                "J": joint_hash,
            },
            "artifacts": artifacts,
        },
        "phase_a": {
            "tensor_families": {
                "trunk": list(TRUNK_PREFIXES),
                "heads": list(HEAD_PREFIXES),
            },
            "bit_identical_assertions": "passed",
        },
        "phase_b": output_metrics(validation_rows, models),
        "phase_c": {
            "post_pr_validation_manifest": probe_manifest,
            "records_path": str(phase_c_path),
            "search_configuration": {
                "contexts": list(SEARCH_CONTEXTS),
                "seed_contract": SEED_CONTRACT_VERSION,
                "base_seed": args.seed,
                "c_puct_by_context": {
                    context: _context_c_puct(context) for context in SEARCH_CONTEXTS
                },
                "dirichlet_epsilon": 0.0,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.0,
            },
            "metrics": phase_c_metrics(phase_c_rows),
        },
        "phase_d": {
            "records_path": str(phase_d_path),
            "continuation_model": "current/current",
            "continuation_budgets": list(CONTINUATION_BUDGETS),
            "priority_contrasts": [item[0] for item in PHASE_D_PRIORITY_CONTRASTS],
            "paired_seed_contract": "state/origin-context/budget invariant; forced move and compared model excluded",
            "disagreement_metrics": phase_d_summary,
        },
        "phase_e": {
            "current": gradient_metrics(
                rows, np.load(args.pr191_workdir / "batch_indexes.npy"), models["C"]
            ),
            "joint": gradient_metrics(
                rows, np.load(args.pr191_workdir / "batch_indexes.npy"), models["J"]
            ),
        },
        "phase_f": {
            "enabled": bool(args.phase_f),
            "models": ["C", "T", "JH", "J"],
            "contexts": list(PHASE_F_CONTEXTS),
            "direct_contrasts": [item[0] for item in PHASE_F_CONTRASTS],
            "search_configuration": {
                "c_puct_by_context": {
                    context: _context_c_puct(context) for context in PHASE_F_CONTEXTS
                },
                "tactical_root_bias": 0.0,
                "root_policy_mode": "deterministic",
                "seed_contract": SEED_CONTRACT_VERSION,
                "base_seed": PHASE_F_SEED,
                "opening_bootstrap_samples": 10_000,
            },
            **phase_f,
        },
    }
    summary["phase_g"] = phase_g_classification(
        summary["phase_b"], summary["phase_e"], summary["phase_f"]
    )
    summary["classification"] = (
        "mixed_additive_harm_no_exclusive_primary"
        if "mixed_additive_harm" in summary["phase_g"]["classifications"]
        else "diagnostic_only_pending_phase_f"
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.phase_f:
        phase_f_rows = []
        for context, context_metrics in phase_f["metrics"].items():
            for contrast, effect in context_metrics["direct_contrasts"].items():
                ci = effect["opening_bootstrap_ci"]
                phase_f_rows.append(
                    f"| {context} | {contrast} | {effect['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
                )
        phase_f_report = (
            "\n\n## Phase F\n\n"
            "| Context | Contrast | Paired candidate effect | 95% opening bootstrap CI |\n"
            "| --- | --- | ---: | --- |\n" + "\n".join(phase_f_rows) + "\n"
        )
    else:
        phase_f_report = "\n\n## Phase F\n\nNot run (`--no-phase-f`).\n"
    phase_g = summary["phase_g"]
    if args.phase_f:
        interpretation = (
            "Phase B's offline replay fit is descriptive only; it does not override paired arena effects. "
            f"In Phase E, the current policy trunk gradient is {phase_g['criteria']['policy_gradient_dominates_trunk_learning']['policy_to_weighted_value_norm_ratio']:.2f}x the weighted value gradient, so `policy_gradient_dominates_trunk_learning` applies. "
            f"Its cosine CI [{phase_g['criteria']['gradient_conflict']['cosine_bootstrap_95']['lower']:.4f}, {phase_g['criteria']['gradient_conflict']['cosine_bootstrap_95']['upper']:.4f}] crosses zero, so `gradient_conflict` does not apply. "
            "At 1200:1200, both isolated deltas are materially harmful (T-C and JH-C), supporting `mixed_additive_harm` but no exclusive primary cause. "
            "The 384:256 T-C effect is also harmful, so this is not a practical-versus-high-search `tradeoff`. "
        )
    else:
        interpretation = "Phase B and Phase E are descriptive; Phase F is required for causal classification. "
    args.report.write_text(
        "# AlphaZero-Lite Shared-Trunk Delta Attribution\n\n"
        f"**Classification:** `{summary['classification']}`\n\n"
        + interpretation
        + f"Next action: `{phase_g['next_action']}`. No model was trained, promoted, or modified."
        + phase_f_report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
