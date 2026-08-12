#!/usr/bin/env python3
# ruff: noqa: E402
"""Matched D384-versus-D1200 AlphaZero visit-policy teacher ablation.

Gameplay is generated once and is immutable before the two offline teacher
searches begin.  The only lane-specific training input is ``policy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    entropy,
    pair_metrics,
    phase_for_state,
    teacher_seed_identity,
)  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    INPUT_ENCODING,
    MODEL_TYPE,
    _hash_state,
    configure_determinism,
    game_split,
    sha256_file,
    write_fixed_npz,
)  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (
    benchmark_budget_results,
    bootstrap_ci,
    build_heldout_suite,
    find_candidate_report,
    pooled_per_opening_differences,
    run_opening_suite_benchmark,
)  # noqa: E402
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (
    changed_key_audit,
    export_checkpoint,
)  # noqa: E402
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    encode_state,
    outcome_for_player,
    policy_from_visits,
    standard_start_state,
)  # noqa: E402
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)  # noqa: E402

EXPECTED_CURRENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"  # noqa: E402
)
TEACHER_BUDGETS = (384, 1200)
POLICY_NAMES = frozenset(
    {
        "policy_hidden_layer.weight",
        "policy_hidden_layer.bias",
        "policy_head.weight",
        "policy_head.bias",
    }
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row).decode("utf-8") + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trajectory_digest(rows: list[dict[str, Any]], winner: int | None) -> str:
    return digest({"states": [row["state"] for row in rows], "winner": winner})


def policy_head_only(model: PolicyValueNet) -> list[str]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in POLICY_NAMES
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if set(names) != POLICY_NAMES:
        raise RuntimeError("residual_v3 policy-head parameter set changed")
    return names


def compare_policy_runs(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    policy_delta = float(
        np.max(np.abs(first["validation_logits"] - second["validation_logits"]))
    )
    value_delta = float(
        np.max(np.abs(first["validation_values"] - second["validation_values"]))
    )
    top1 = float(
        np.mean(
            np.argmax(first["validation_logits"], 1)
            == np.argmax(second["validation_logits"], 1)
        )
    )
    exact = first["parameter_sha256"] == second["parameter_sha256"]
    return {
        "exact_parameter_hashes": exact,
        "checkpoint_hashes_identical": first["checkpoint_sha256"]
        == second["checkpoint_sha256"],
        "maximum_policy_prediction_difference": policy_delta,
        "maximum_value_prediction_difference": value_delta,
        "policy_top1_agreement": top1,
        "passes": exact or (policy_delta <= 1e-7 and top1 == 1.0),
    }


def teacher_target(
    *,
    evaluator: CheckpointEvaluator,
    state: dict[str, Any],
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal = [int(move) for move in game.possible_moves()]
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(seed),
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        normalize_values=False,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    policy = np.asarray(policy_from_visits(visits, legal, temperature=1.0), dtype=float)
    order = sorted(
        legal,
        key=lambda move: (
            -int(root.children[move].visit_count),
            -float(root.children[move].q_value),
            -float(root.children[move].prior),
            move,
        ),
    )
    return {
        "policy": policy.tolist(),
        "legal_moves": legal,
        "visit_counts": [int(value) for value in visits],
        "top_move": int(order[0]),
        "entropy": entropy(policy),
        "root_value": float(root.q_value),
        "top1_top2_visit_margin": float(policy[order[0]])
        - (float(policy[order[1]]) if len(order) > 1 else 0.0),
        "target_search_profile": {
            "simulations": simulations,
            "c_puct": 1.25,
            "tactical_root_bias": 0.0,
            "dirichlet_epsilon": 0.0,
            "root_policy": "visits",
        },
        "target_hash": digest(policy.tolist()),
    }


def generate_trajectory_corpus(
    *, checkpoint: Path, games: int, seed: int, rows_target: int
) -> list[dict[str, Any]]:
    """Generate the sole D384 gameplay corpus, retaining raw states and moves."""
    evaluator = CheckpointEvaluator(checkpoint, input_encoding=INPUT_ENCODING)
    rows: list[dict[str, Any]] = []
    for game_id in range(games):
        game = KalahGame.from_state(standard_start_state())
        rng = random.Random((seed * 1_000_003) + game_id)
        game_rows: list[dict[str, Any]] = []
        for ply in range(200):
            if game.over():
                break
            state = game.to_state()
            legal = game.possible_moves()
            if not legal:
                break
            search = PUCT(
                evaluator=evaluator,
                simulations=384,
                c_puct=1.25,
                rng=rng,
                root_policy_mode="visit_count",
                tactical_root_bias=0.0,
            )
            visits, _root = search.run(
                game,
                dirichlet_alpha=0.3 if ply < 8 else None,
                dirichlet_epsilon=0.25 if ply < 8 else 0.0,
            )
            temperature = 0.67 if ply < 8 else 0.0
            move = (
                int(max(legal, key=lambda item: (visits[item], -item)))
                if temperature <= 0.05
                else _sample(policy_from_visits(visits, legal, temperature), legal, rng)
            )
            game_rows.append(
                {
                    "game_id": game_id,
                    "game_index": game_id,
                    "ply": ply,
                    "move_index": ply,
                    "raw_state": state,
                    "state": encode_state(state, input_encoding=INPUT_ENCODING),
                    "player": int(game.current_player),
                    "chosen_gameplay_move": move,
                    "phase": phase_for_state(state),
                }
            )
            if not game.move(game.pit_index(move)):
                raise RuntimeError("gameplay selected an illegal move")
        if not game.over():
            raise RuntimeError("self-play did not terminate within 200 plies")
        winner = game.winner
        trace_hash = trajectory_digest(game_rows, winner)
        for row in game_rows:
            row["winner"] = winner
            row["terminal_outcome_target"] = outcome_for_player(winner, row["player"])
            row["value"] = row["terminal_outcome_target"]
            row["trajectory_hash"] = trace_hash
        rows.extend(game_rows)
        if len(rows) >= rows_target:
            break
    if len(rows) < rows_target:
        raise RuntimeError(f"only generated {len(rows)} rows; requested {rows_target}")
    return rows


def _sample(policy: list[float], legal: list[int], rng: random.Random) -> int:
    threshold, total = rng.random(), 0.0
    for move in legal:
        total += float(policy[move])
        if threshold <= total:
            return int(move)
    return int(legal[-1])


def retarget(
    rows: list[dict[str, Any]], *, checkpoint: Path, seed: int
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    evaluator = CheckpointEvaluator(checkpoint, input_encoding=INPUT_ENCODING)
    lanes: dict[int, list[dict[str, Any]]] = {budget: [] for budget in TEACHER_BUDGETS}
    telemetry: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in TEACHER_BUDGETS
    }
    for row in rows:
        search_seed, context_hash = teacher_seed_identity(
            state_hash=digest(row["raw_state"]), experiment_seed=seed
        )
        for budget in TEACHER_BUDGETS:
            target = teacher_target(
                evaluator=evaluator,
                state=row["raw_state"],
                simulations=budget,
                seed=search_seed,
            )
            replay = {key: value for key, value in row.items() if key != "raw_state"}
            replay.update(
                {
                    "policy": target["policy"],
                    "policy_teacher": f"D{budget}",
                    "policy_teacher_seed_context_hash": context_hash,
                    "policy_teacher_telemetry": target,
                }
            )
            lanes[budget].append(replay)
            telemetry[budget].append(target)
    return lanes, telemetry


SCIENTIFIC_FIELDS = (
    "game_id",
    "game_index",
    "ply",
    "move_index",
    "state",
    "player",
    "chosen_gameplay_move",
    "winner",
    "terminal_outcome_target",
    "value",
    "trajectory_hash",
)


def paired_invariants(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    checks = {"row_count": len(left) == len(right)}
    checks.update(
        {
            field: len(left) == len(right)
            and all(a.get(field) == b.get(field) for a, b in zip(left, right))
            for field in SCIENTIFIC_FIELDS
        }
    )
    non_policy_equal = all(
        canonical(
            {
                k: v
                for k, v in a.items()
                if k not in {"policy", "policy_teacher", "policy_teacher_telemetry"}
            }
        )
        == canonical(
            {
                k: v
                for k, v in b.items()
                if k not in {"policy", "policy_teacher", "policy_teacher_telemetry"}
            }
        )
        for a, b in zip(left, right)
    )
    hashes = {
        "state_sequence": digest([row["state"] for row in left]),
        "outcome_targets": digest([row["terminal_outcome_target"] for row in left]),
        "value_targets": digest([row["value"] for row in left]),
        "D384_policies": digest([row["policy"] for row in left]),
        "D1200_policies": digest([row["policy"] for row in right]),
    }
    return {
        "passes": all(checks.values()) and non_policy_equal,
        "checks": checks,
        "non_policy_byte_equivalent": non_policy_equal,
        "hashes": hashes,
    }


def target_audit(
    rows: list[dict[str, Any]], telemetry: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    records = []
    for row, low, high in zip(rows, telemetry[384], telemetry[1200]):
        records.append(
            {
                **row,
                "current_policy_entropy": entropy(low["policy"]),
                **pair_metrics(low, high),
                "margin": low["top1_top2_visit_margin"],
            }
        )

    def means(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(values),
            "mean_js": float(np.mean([r["js_divergence"] for r in values]))
            if values
            else 0.0,
            "top1_disagreement": float(np.mean([r["top_move_change"] for r in values]))
            if values
            else 0.0,
            "top2_agreement": float(np.mean([r["top2_set_agreement"] for r in values]))
            if values
            else 0.0,
            "rank_correlation": float(np.mean([r["spearman"] for r in values]))
            if values
            else 0.0,
            "entropy_difference": float(np.mean([r["entropy_delta"] for r in values]))
            if values
            else 0.0,
        }

    result = {
        "global": means(records),
        "by_player": {},
        "by_phase": {},
        "by_policy_entropy_quartile": {},
        "by_visit_margin_quartile": {},
    }
    for name, key in (("by_player", "player"), ("by_phase", "phase")):
        result[name] = {
            str(value): means([r for r in records if r[key] == value])
            for value in sorted({r[key] for r in records}, key=str)
        }
    for name, field in (
        ("by_policy_entropy_quartile", "current_policy_entropy"),
        ("by_visit_margin_quartile", "margin"),
    ):
        cutoffs = np.quantile([r[field] for r in records], [0.25, 0.5, 0.75])
        result[name] = {}
        for index in range(4):
            low = -np.inf if index == 0 else cutoffs[index - 1]
            high = np.inf if index == 3 else cutoffs[index]
            result[name][str(index + 1)] = means(
                [r for r in records if r[field] > low and r[field] <= high]
            )
    return result


def build_manifest(
    rows: list[dict[str, Any]], workdir: Path, current: Path, seed: int
) -> dict[str, Any]:
    train_games, validation_games = game_split(rows, seed)
    train = np.asarray(
        [i for i, row in enumerate(rows) if row["game_id"] in set(train_games)],
        dtype=np.int64,
    )
    validation = np.asarray(
        [i for i, row in enumerate(rows) if row["game_id"] in set(validation_games)],
        dtype=np.int64,
    )
    plan = np.arange(len(train), dtype=np.int64)
    rng = np.random.default_rng(seed)
    plan = rng.permutation(plan)
    batches = np.full(((len(plan) + 511) // 512, 512), -1, dtype=np.int64)
    batches.flat[: len(plan)] = plan
    init = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "initialization.npz"
    )
    for name, value in (
        ("train_source_indexes.npy", train),
        ("validation_source_indexes.npy", validation),
        ("batch_indexes.npy", batches),
    ):
        np.save(workdir / name, value, allow_pickle=False)
    return {
        "train_games": train_games,
        "validation_games": validation_games,
        "game_membership_sha256": digest(
            {"train": train_games, "validation": validation_games}
        ),
        "initialization_checkpoint": str(init),
        "batch_plan_sha256": sha256_file(workdir / "batch_indexes.npy"),
        "optimizer": {"type": "Adam", "lr": 1e-5},
        "epochs": 1,
        "batch_size": 512,
        "gradient_clip": 1.0,
        "seed": seed,
    }


def train_lane(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    workdir: Path,
    label: str,
    device: torch.device,
) -> dict[str, Any]:
    configure_determinism(device, manifest["seed"])
    source = np.load(workdir / "train_source_indexes.npy", allow_pickle=False)
    plan = np.load(workdir / "batch_indexes.npy", allow_pickle=False)
    selected = [rows[int(index)] for index in source]
    x = np.asarray([row["state"] for row in selected], dtype=np.float32)
    p = np.asarray([row["policy"] for row in selected], dtype=np.float32)
    masks = legal_mask_matrix_for_encoded_states(x)
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    load_checkpoint_into_model(model, Path(manifest["initialization_checkpoint"]))
    trainable = policy_head_only(model)
    model.to(device).train()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5
    )
    tensors = [torch.from_numpy(value).to(device) for value in (x, p, masks)]
    policy_loss = torch.zeros((), device=device)
    for batch in plan:
        indexes = torch.as_tensor(batch[batch >= 0], device=device, dtype=torch.long)
        logits, _value = model(tensors[0][indexes])
        policy_loss = compute_policy_cross_entropy(
            logits.masked_fill(tensors[2][indexes] <= 0, -1e9), tensors[1][indexes]
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
    output = workdir / label
    output.mkdir(exist_ok=True)
    checkpoint = output / "checkpoint.npz"
    checkpoint_sha = write_fixed_npz(checkpoint, checkpoint_from_model(model))
    artifact = output / "artifact"
    export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=artifact,
        version=label,
        policy_loss=float(policy_loss.item()),
        value_loss=0.0,
    )
    validation = np.load(workdir / "validation_source_indexes.npy", allow_pickle=False)
    with torch.no_grad():
        logits, values = model(
            torch.from_numpy(
                np.asarray(
                    [rows[int(i)]["state"] for i in validation], dtype=np.float32
                )
            ).to(device)
        )
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "artifact": str(artifact),
        "parameter_sha256": _hash_state(model),
        "validation_logits": logits.cpu().numpy(),
        "validation_values": values.cpu().numpy(),
        "trainable_parameter_names": trainable,
        "policy_loss": float(policy_loss.item()),
    }


def cross_teacher_validation(
    rows: list[dict[str, Any]],
    indexes: np.ndarray,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    targets = {
        name: np.asarray([rows[int(i)]["policy"] for i in indexes], dtype=float)
        for name, rows in candidates.pop("_targets").items()
    }
    # This function is intentionally fed only raw logits; masking is represented in target support.
    result = {}
    for name, run in candidates.items():
        probs = np.exp(
            run["validation_logits"]
            - np.max(run["validation_logits"], axis=1, keepdims=True)
        )
        probs /= probs.sum(axis=1, keepdims=True)
        result[name] = {}
        for teacher, target in targets.items():
            ce = -np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1).mean()
            midpoint = (target + probs) / 2
            js = (
                (
                    target
                    * np.log(np.maximum(target, 1e-12) / np.maximum(midpoint, 1e-12))
                ).sum(1)
                + (
                    probs
                    * np.log(np.maximum(probs, 1e-12) / np.maximum(midpoint, 1e-12))
                ).sum(1)
            ).mean() / 2
            result[name][teacher] = {
                "ce": float(ce),
                "js": float(js),
                "top1_agreement": float(
                    np.mean(np.argmax(probs, 1) == np.argmax(target, 1))
                ),
            }
    return result


def initial_validation_outputs(
    rows: list[dict[str, Any]],
    indexes: np.ndarray,
    checkpoint: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    load_checkpoint_into_model(model, checkpoint)
    model.to(device).eval()
    states = torch.from_numpy(
        np.asarray([rows[int(index)]["state"] for index in indexes], dtype=np.float32)
    ).to(device)
    with torch.no_grad():
        logits, values = model(states)
    return logits.cpu().numpy(), values.cpu().numpy()


def raw_policy_changes(
    *,
    current_logits: np.ndarray,
    candidate_logits: dict[str, np.ndarray],
    teacher_384: np.ndarray,
    teacher_1200: np.ndarray,
) -> dict[str, Any]:
    def probabilities(logits: np.ndarray) -> np.ndarray:
        result = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return result / result.sum(axis=1, keepdims=True)

    current = probabilities(current_logits)
    disagreement = np.argmax(teacher_384, 1) != np.argmax(teacher_1200, 1)
    report = {}
    for name, logits in candidate_logits.items():
        policy = probabilities(logits)
        changed = np.argmax(policy, 1) != np.argmax(current, 1)
        kl = np.sum(
            policy * np.log(np.maximum(policy, 1e-12) / np.maximum(current, 1e-12)),
            axis=1,
        )
        entropy_delta = -np.sum(
            policy * np.log(np.maximum(policy, 1e-12)), axis=1
        ) + np.sum(current * np.log(np.maximum(current, 1e-12)), axis=1)
        d1200_minus_d384 = 0.0
        if disagreement.any():
            d1200_minus_d384 = float(
                np.mean(
                    (
                        np.argmax(policy[disagreement], 1)
                        == np.argmax(teacher_1200[disagreement], 1)
                    ).astype(float)
                    - (
                        np.argmax(policy[disagreement], 1)
                        == np.argmax(teacher_384[disagreement], 1)
                    ).astype(float)
                )
            )
        report[name] = {
            "raw_top1_changed_rate": float(changed.mean()),
            "policy_kl_from_current": float(kl.mean()),
            "entropy_change": float(entropy_delta.mean()),
            "teacher_D384_top1_agreement": float(
                np.mean(np.argmax(policy, 1) == np.argmax(teacher_384, 1))
            ),
            "teacher_D1200_top1_agreement": float(
                np.mean(np.argmax(policy, 1) == np.argmax(teacher_1200, 1))
            ),
            "on_teacher_disagreement": {
                "rows": int(disagreement.sum()),
                "D1200_minus_D384_teacher_agreement": d1200_minus_d384,
            },
        }
    return report


def screen(
    *,
    workdir: Path,
    suite: Path,
    current: Path,
    d384: Path,
    d1200: Path,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    # The opening benchmark keys reports by directory name, so aliases prevent
    # both exported ``artifact`` directories from collapsing into one candidate.
    aliases = {}
    for name, target in (("D384", d384), ("D1200", d1200)):
        alias = workdir / name
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.symlink(target.resolve(), alias, target_is_directory=True)
        aliases[name] = alias
    report = run_opening_suite_benchmark(
        workdir=str(workdir),
        suite=str(suite),
        current=str(current),
        candidates=",".join(
            str(path) for path in (current, aliases["D384"], aliases["D1200"])
        ),
        budget_pairs="384:256,768:256,768:768,1200:1200,1200:256,256:768",
        games_per_opening=2,
        seed=seed,
        workers=workers,
        timeout=14400,
        seed_contract="azlite_eval_seed_v2",
        seed_ledger_output=str(workdir / "seed_ledger.jsonl"),
    )
    labels = {
        "current": current.name,
        "D384": aliases["D384"].name,
        "D1200": aliases["D1200"].name,
    }
    reports = {
        name: find_candidate_report(report, label) for name, label in labels.items()
    }
    if any(value is None for value in reports.values()):
        raise RuntimeError("paired benchmark did not report every candidate")
    budgets = {
        name: benchmark_budget_results(value)
        for name, value in reports.items()
        if value is not None
    }
    rows = {
        "screen": {
            "candidates": {
                name: {"budget_results": values} for name, values in budgets.items()
            }
        }
    }
    comparisons = {}
    for name, left, right in (
        ("D1200_minus_D384", "D1200", "D384"),
        ("D1200_minus_current", "D1200", "current"),
        ("D384_minus_current", "D384", "current"),
    ):
        comparisons[name] = {
            budget: bootstrap_ci(
                pooled_per_opening_differences(
                    suite_rows=rows,
                    candidate_a=left,
                    candidate_b=right,
                    budget_pair=budget,
                    metric_key="ds",
                ),
                seed=seed,
            )
            for budget in budgets["D1200"]
        }
    return {
        "report_sha256": sha256_file(workdir / "temperature_benchmark_report.json"),
        "budgets": budgets,
        "paired_opening_bootstrap_95": comparisons,
    }


def medium_pass(result: dict[str, Any]) -> bool:
    d1200_d384 = result["paired_opening_bootstrap_95"]["D1200_minus_D384"]
    d1200_current = result["paired_opening_bootstrap_95"]["D1200_minus_current"]
    return bool(
        d1200_d384["384:256"]["mean"] >= 0.03
        and d1200_d384["384:256"]["lower"] > 0
        and d1200_current["384:256"]["mean"] > 0
        and all(
            d1200_current[key]["mean"] >= -0.03
            for key in ("768:768", "1200:1200", "1200:256")
        )
    )


def fixed_pass(result: dict[str, Any]) -> bool:
    values = result["paired_opening_bootstrap_95"]["D1200_minus_current"]
    return bool(
        values["384:256"]["mean"] >= 0.05
        and values["384:256"]["lower"] > 0.01
        and values["768:768"]["mean"] >= -0.05
        and all(values[key]["mean"] >= -0.03 for key in ("1200:1200", "1200:256"))
    )


def hierarchical_heldout(
    screens: dict[str, dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    budgets = ("384:256", "768:768", "1200:1200", "1200:256")
    results: dict[str, Any] = {}
    for budget in budgets:
        blocks = []
        for result in screens.values():
            candidates = result["budgets"]
            left = {
                entry["opening_prefix"]: entry["ds"]
                for entry in candidates["D1200"][budget]["per_opening_metrics"]
            }
            right = {
                entry["opening_prefix"]: entry["ds"]
                for entry in candidates["current"][budget]["per_opening_metrics"]
            }
            blocks.append(
                np.asarray(
                    [
                        float(left[key]) - float(right[key])
                        for key in sorted(set(left) & set(right))
                    ],
                    dtype=float,
                )
            )
        rng = np.random.default_rng(seed)
        draws = [
            float(
                rng.choice(
                    blocks[rng.integers(len(blocks))], size=len(blocks[0]), replace=True
                ).mean()
            )
            for _ in range(10_000)
        ]
        suite_means = [float(block.mean()) for block in blocks]
        results[budget] = {
            "mean": float(np.mean(suite_means)),
            "lower": float(np.percentile(draws, 2.5)),
            "upper": float(np.percentile(draws, 97.5)),
            "suite_count": len(blocks),
            "worst_suite": min(suite_means),
        }
    passes = bool(
        results["384:256"]["mean"] >= 0.05
        and results["384:256"]["lower"] > 0.01
        and results["768:768"]["mean"] >= -0.05
        and results["1200:1200"]["mean"] >= -0.03
        and results["1200:256"]["mean"] >= -0.03
    )
    return {
        "method": "hierarchical_suite_then_opening",
        "budgets": results,
        "passes": passes,
    }


def classify(
    *,
    invariants: dict[str, Any],
    reproducible: bool,
    target_audit: dict[str, Any] | None = None,
    medium: dict[str, Any] | None = None,
    fixed_large: dict[str, Any] | None = None,
    heldout: dict[str, Any] | None = None,
) -> str:
    if not invariants["passes"]:
        return "paired_teacher_ablation_invalid"
    if not reproducible:
        return "full_scale_training_nondeterministic"
    if medium is None:
        return "stronger_policy_teacher_experiment_incomplete"
    deltas = medium["paired_opening_bootstrap_95"]
    d1200_d384 = deltas["D1200_minus_D384"]
    d1200_current = deltas["D1200_minus_current"]
    d384_current = deltas["D384_minus_current"]
    measurable = bool(
        target_audit and target_audit["global"]["top1_disagreement"] > 0.0
    )
    if (
        d1200_current["384:256"]["mean"] < -0.03
        and d384_current["384:256"]["mean"] < -0.03
    ):
        return "policy_head_update_remains_primary_failure"
    if (
        d1200_d384["384:256"]["mean"] < 0
        and d1200_current["384:256"]["mean"] < d384_current["384:256"]["mean"]
    ):
        return "stronger_teacher_overfits_search"
    if not medium_pass(medium):
        if (
            measurable
            and d1200_d384["384:256"]["lower"] <= 0 <= d1200_d384["384:256"]["upper"]
        ):
            return "stronger_teacher_targets_do_not_improve_learning"
        if d1200_d384["384:256"]["lower"] > 0:
            return "stronger_teacher_improves_learning_but_not_strength"
        return "stronger_teacher_targets_do_not_improve_learning"
    if fixed_large is None or not fixed_pass(fixed_large):
        return "stronger_teacher_improves_learning_but_not_strength"
    if heldout is None or not heldout.get("passes", False):
        return "stronger_teacher_improves_learning_but_not_strength"
    return "stronger_policy_teacher_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_stronger_policy_teacher")
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256", default=EXPECTED_CURRENT_SHA256
    )
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--rows-target", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument(
        "--heldout-seeds", default="43,44,45,46,47,48,49,50,51,52,53,54"
    )
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir, current = Path(args.workdir), Path(args.current)
    workdir.mkdir(parents=True, exist_ok=True)
    if sha256_file(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    initialization = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "current.npz"
    )
    corpus_path = workdir / "trajectory_replay.jsonl"
    if corpus_path.exists():
        corpus = read_jsonl(corpus_path)
    else:
        corpus = generate_trajectory_corpus(
            checkpoint=initialization,
            games=args.games,
            seed=args.seed,
            rows_target=args.rows_target,
        )
        write_jsonl(corpus_path, corpus)
    trajectory_manifest = {
        "schema": "azlite_stronger_policy_teacher_trajectory_v1",
        "rows": len(corpus),
        "games": len({row["game_id"] for row in corpus}),
        "trajectory_replay_sha256": sha256_file(corpus_path),
        "gameplay": {
            "simulations": 384,
            "c_puct": 1.25,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature": 0.67,
            "temperature_late": 0.0,
            "temperature_threshold": 8,
        },
    }
    write_json(workdir / "trajectory_manifest.json", trajectory_manifest)
    paths = {budget: workdir / f"replay_D{budget}.jsonl" for budget in TEACHER_BUDGETS}
    if all(path.is_file() for path in paths.values()):
        lanes = {budget: read_jsonl(path) for budget, path in paths.items()}
        telemetry = {
            budget: [row["policy_teacher_telemetry"] for row in lanes[budget]]
            for budget in TEACHER_BUDGETS
        }
    else:
        lanes, telemetry = retarget(corpus, checkpoint=initialization, seed=args.seed)
        for budget in TEACHER_BUDGETS:
            write_jsonl(paths[budget], lanes[budget])
    manifest = build_manifest(lanes[384], workdir, current, args.seed)
    invariants = paired_invariants(lanes[384], lanes[1200])
    invariants["hashes"]["game_membership"] = manifest["game_membership_sha256"]
    audit = target_audit(corpus, telemetry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {}
    for budget in TEACHER_BUDGETS:
        runs[f"D{budget}_run_a"] = train_lane(
            lanes[budget],
            manifest,
            workdir,
            f"d{budget}_policy_teacher_e1_run_a",
            device,
        )
        runs[f"D{budget}_run_b"] = train_lane(
            lanes[budget],
            manifest,
            workdir,
            f"d{budget}_policy_teacher_e1_run_b",
            device,
        )
    reproduction = {
        f"D{budget}": compare_policy_runs(
            runs[f"D{budget}_run_a"], runs[f"D{budget}_run_b"]
        )
        for budget in TEACHER_BUDGETS
    }
    freeze = {
        f"D{budget}": changed_key_audit(
            reference_checkpoint=Path(manifest["initialization_checkpoint"]),
            candidate_checkpoint=Path(runs[f"D{budget}_run_a"]["checkpoint"]),
            allowed_families={"policy"},
        )
        for budget in TEACHER_BUDGETS
    }
    validation = np.load(workdir / "validation_source_indexes.npy", allow_pickle=False)
    cross = cross_teacher_validation(
        lanes[384],
        validation,
        {
            "D384": runs["D384_run_a"],
            "D1200": runs["D1200_run_a"],
            "_targets": {"D384": lanes[384], "D1200": lanes[1200]},
        },
    )
    initial_logits, initial_values = initial_validation_outputs(
        lanes[384], validation, Path(manifest["initialization_checkpoint"]), device
    )
    value_head_exact = {
        f"D{budget}": bool(
            np.array_equal(
                initial_values, runs[f"D{budget}_run_a"]["validation_values"]
            )
        )
        for budget in TEACHER_BUDGETS
    }
    if not all(value_head_exact.values()):
        raise RuntimeError("policy-only training changed value-head predictions")
    raw_changes = raw_policy_changes(
        current_logits=initial_logits,
        candidate_logits={
            f"D{budget}": runs[f"D{budget}_run_a"]["validation_logits"]
            for budget in TEACHER_BUDGETS
        },
        teacher_384=np.asarray(
            [lanes[384][int(index)]["policy"] for index in validation]
        ),
        teacher_1200=np.asarray(
            [lanes[1200][int(index)]["policy"] for index in validation]
        ),
    )
    reproducible = all(value["passes"] for value in reproduction.values()) and all(
        value["passes"] for value in freeze.values()
    )
    medium = (
        screen(
            workdir=workdir / "medium",
            suite=Path(args.medium_suite),
            current=current,
            d384=Path(runs["D384_run_a"]["artifact"]),
            d1200=Path(runs["D1200_run_a"]["artifact"]),
            seed=args.seed,
            workers=args.workers,
        )
        if reproducible and invariants["passes"]
        else None
    )
    fixed_large = (
        screen(
            workdir=workdir / "fixed_large",
            suite=Path(args.fixed_large_suite),
            current=current,
            d384=Path(runs["D384_run_a"]["artifact"]),
            d1200=Path(runs["D1200_run_a"]["artifact"]),
            seed=args.seed,
            workers=args.workers,
        )
        if medium is not None and medium_pass(medium)
        else None
    )
    heldout_screens = {}
    if fixed_large is not None and fixed_pass(fixed_large):
        for heldout_seed in (
            int(value) for value in args.heldout_seeds.split(",") if value
        ):
            heldout_path = workdir / "heldout" / f"suite_{heldout_seed}.jsonl"
            build_heldout_suite(
                fixed_suite_path=Path(args.fixed_large_suite),
                seed=heldout_seed,
                out_path=heldout_path,
            )
            heldout_screens[str(heldout_seed)] = screen(
                workdir=workdir / "heldout" / str(heldout_seed),
                suite=heldout_path,
                current=current,
                d384=Path(runs["D384_run_a"]["artifact"]),
                d1200=Path(runs["D1200_run_a"]["artifact"]),
                seed=args.seed,
                workers=args.workers,
            )
    heldout = (
        hierarchical_heldout(heldout_screens, seed=args.seed)
        if heldout_screens
        else None
    )
    summary = {
        "schema": "azlite_stronger_policy_teacher_ablation_v1",
        "seed_contract": "azlite_eval_seed_v2",
        "current_weights_sha256": args.expected_current_weights_sha256,
        "trajectory_corpus_manifest": trajectory_manifest,
        "paired_data_invariants": invariants,
        "target_differences": audit,
        "training_manifest": manifest,
        "deterministic_training": {
            key: {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
            for key, value in runs.items()
        },
        "reproducibility": reproduction,
        "freeze_audit": freeze,
        "value_head_predictions_exact_current": value_head_exact,
        "cross_teacher_validation": cross,
        "raw_policy_changes": raw_changes,
        "medium_strength": medium,
        "fixed_large": fixed_large,
        "heldout": heldout,
        "heldout_screens": heldout_screens,
        "classification": classify(
            invariants=invariants,
            reproducible=reproducible,
            target_audit=audit,
            medium=medium,
            fixed_large=fixed_large,
            heldout=heldout,
        ),
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT / "docs/data/alphazero-lite-stronger-policy-teacher-summary.json",
        summary,
    )
    (REPO_ROOT / "docs/alphazero-lite-stronger-policy-teacher-results.md").write_text(
        "# Stronger Policy Teacher Ablation\n\n```json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"classification": summary["classification"], "rows": len(corpus)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
