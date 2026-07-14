#!/usr/bin/env python3
"""Auditable deterministic current-initialized joint-head update.

This deliberately does not call :func:`train.train`: that API derives its own
split and permutation.  The persisted manifest is the sole authority for both.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    build_policy_probe_rows,
    build_search_probe_rows,
    changed_key_audit,
    evaluate_search_outputs,
    evaluate_search_probe_metrics,
    export_checkpoint,
)
from ml.alphazero_lite.run_current_init_policy_only_distillation_preflight import (  # noqa: E402
    evaluate_policy_probe_metrics,
    evaluate_raw_outputs,
)
from ml.alphazero_lite.cpuct_schedule import parse_cpuct_schedule_json  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    benchmark_budget_results,
    bootstrap_ci,
    find_candidate_report,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

MODEL_TYPE = "residual_v3"
INPUT_ENCODING = "kalah_v3"
HIDDEN_SIZES = (96, 3)
HEAD_NAMES = frozenset(
    {
        "policy_hidden_layer.weight",
        "policy_hidden_layer.bias",
        "policy_head.weight",
        "policy_head.bias",
        "value_hidden_layer.weight",
        "value_hidden_layer.bias",
        "value_head.weight",
        "value_head.bias",
    }
)
PR155_LR = 1e-5
PR155_VALUE_LOSS_WEIGHT = 0.6
CAPTURE_STEPS = frozenset({1, 10, 50})


def sha256_file(path: Path) -> str:
    """Return a file's SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a byte payload's SHA256 digest."""
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical, reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash the complete manifest other than its self-referential digest."""
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256_excluding_this_field"
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    """Reload and verify every immutable input and generated manifest file."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.get("manifest_sha256_excluding_this_field")
    if expected_manifest_hash != canonical_manifest_hash(manifest):
        raise RuntimeError("training manifest hash mismatch")
    for path_text, expected_hash in manifest.get("files", {}).items():
        path = Path(path_text)
        if not path.is_file():
            raise RuntimeError(f"manifest referenced file is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"manifest referenced file hash mismatch: {path}")
    return manifest


def fixed_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize checkpoint arrays without ZIP timestamps or insertion ordering."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array = io.BytesIO()
            np.lib.format.write_array(
                array, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array.getvalue())
    return output.getvalue()


def write_fixed_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write a deterministic checkpoint and return its digest."""
    payload = fixed_npz_bytes(arrays)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read source rows without changing their order."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def configure_determinism(device: torch.device, seed: int) -> dict[str, Any]:
    """Apply all relevant deterministic controls before model construction."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "device": str(device),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def freeze_joint_heads(model: PolicyValueNet) -> list[str]:
    """Freeze the complete trunk and return the only allowed trainable names."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in HEAD_NAMES
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if set(names) != HEAD_NAMES:
        raise RuntimeError("residual_v3 joint head parameter set changed")
    return names


def _game_strata(rows: list[dict[str, Any]]) -> dict[int, tuple[str, str, int]]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_index"])].append(row)
    result = {}
    for game, game_rows in by_game.items():
        outcome = int(np.sign(np.mean([float(row["value"]) for row in game_rows])))
        seat = str(
            game_rows[0].get("seat_context", f"player_{game_rows[0].get('player', 0)}")
        )
        result[game] = (
            str(outcome),
            seat,
            int(game_rows[0].get("game_length", len(game_rows)) // 10),
        )
    return result


def game_split(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    """Make a seeded 85:15 game split, retaining outcome/seat/length strata."""
    strata: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for game, key in _game_strata(rows).items():
        strata[key].append(game)
    rng = random.Random(seed)
    train, validation = [], []
    for games in strata.values():
        rng.shuffle(games)
        count = int(round(len(games) * 0.85))
        if len(games) > 1:
            count = min(len(games) - 1, max(1, count))
        train.extend(games[:count])
        validation.extend(games[count:])
    if not validation and len(train) > 1:
        validation.append(train.pop())
    return sorted(train), sorted(validation)


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    workdir: Path,
    current: Path,
    replay: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    replay_audit: Path,
) -> dict[str, Any]:
    """Persist immutable game membership, source positions, and epoch batches."""
    train_games, validation_games = game_split(rows, seed)
    train_set, validation_set = set(train_games), set(validation_games)
    train_indexes = np.asarray(
        [i for i, row in enumerate(rows) if int(row["game_index"]) in train_set],
        dtype=np.int64,
    )
    validation_indexes = np.asarray(
        [i for i, row in enumerate(rows) if int(row["game_index"]) in validation_set],
        dtype=np.int64,
    )
    if not len(train_indexes) or not len(validation_indexes):
        raise ValueError("game split must contain training and validation rows")
    rng = np.random.default_rng(seed)
    plan = np.concatenate([rng.permutation(len(train_indexes)) for _ in range(epochs)])
    plan = (
        plan.reshape(-1, batch_size)
        if len(plan) % batch_size == 0
        else np.array_split(plan, range(batch_size, len(plan), batch_size))
    )
    # Object arrays are forbidden. A short final batch is represented by -1 padding.
    if isinstance(plan, list):
        packed = np.full((len(plan), batch_size), -1, dtype=np.int64)
        for index, batch in enumerate(plan):
            packed[index, : len(batch)] = batch
        plan = packed
    init = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "initialization.npz"
    )
    paths = {
        "initialization_checkpoint": init,
        "train_game_ids": workdir / "train_game_ids.json",
        "validation_game_ids": workdir / "validation_game_ids.json",
        "train_source_indexes": workdir / "train_source_indexes.npy",
        "validation_source_indexes": workdir / "validation_source_indexes.npy",
        "batch_indexes": workdir / "batch_indexes.npy",
        "replay": replay,
        "replay_audit": replay_audit,
        "current_weights": current / "weights.json",
    }
    paths["train_game_ids"].write_text(json.dumps(train_games) + "\n", encoding="utf-8")
    paths["validation_game_ids"].write_text(
        json.dumps(validation_games) + "\n", encoding="utf-8"
    )
    np.save(paths["train_source_indexes"], train_indexes, allow_pickle=False)
    np.save(paths["validation_source_indexes"], validation_indexes, allow_pickle=False)
    np.save(paths["batch_indexes"], plan, allow_pickle=False)
    manifest = {
        "schema": "azlite_deterministic_joint_heads_manifest_v1",
        "candidate": "deterministic_joint_heads_e1",
        "replay_path": str(replay),
        "replay_audit_path": str(replay_audit),
        "current_weights_sha256": sha256_file(current / "weights.json"),
        "initialization_checkpoint_sha256": sha256_file(init),
        "replay_sha256": sha256_file(replay),
        "replay_rows": len(rows),
        "replay_audit_sha256": sha256_file(replay_audit),
        "train_game_ids": train_games,
        "validation_game_ids": validation_games,
        "train_source_row_indexes": train_indexes.tolist(),
        "validation_source_row_indexes": validation_indexes.tolist(),
        "batch_plan": {"epochs": epochs, "batch_size": batch_size, "padding": -1},
        "architecture": {
            "model_type": MODEL_TYPE,
            "hidden_sizes": list(HIDDEN_SIZES),
            "input_encoding": INPUT_ENCODING,
        },
        "trainable_parameter_names": sorted(HEAD_NAMES),
        "optimizer": {"type": "Adam", "lr": PR155_LR, "weight_decay": 0.0},
        "policy_loss": "legal_mask_aware_cross_entropy",
        "value_loss": "huber",
        "value_loss_weight": PR155_VALUE_LOSS_WEIGHT,
        "huber_delta": 1.0,
        "gradient_clip": 1.0,
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "export": {
            "model_type": MODEL_TYPE,
            "input_encoding": INPUT_ENCODING,
            "rules_version": "kalah_v3",
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "files": {str(path): sha256_file(path) for path in paths.values()},
    }
    manifest["batch_plan_sha256"] = sha256_file(paths["batch_indexes"])
    manifest["manifest_sha256_excluding_this_field"] = canonical_manifest_hash(manifest)
    manifest_path = workdir / "training_manifest.json"
    write_json(manifest_path, manifest)
    return verify_manifest(manifest_path)


def _hash_state(model: PolicyValueNet) -> dict[str, str]:
    return {
        name: sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())
        for name, value in model.state_dict().items()
    }


def _optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    """Hash optimizer state without nondeterministic pickle serialization."""
    state: list[tuple[int, str, str]] = []
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        for name, value in sorted(optimizer.state.get(parameter, {}).items()):
            encoded = (
                value.detach().cpu().contiguous().numpy().tobytes().hex()
                if isinstance(value, torch.Tensor)
                else repr(value)
            )
            state.append((index, str(name), encoded))
    return sha256_bytes(json.dumps(state, separators=(",", ":")).encode())


def _capture(model: PolicyValueNet, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Capture all required numerical identities at a fixed optimizer boundary."""
    return {
        "parameter_sha256": _hash_state(model),
        "optimizer_state_sha256": _optimizer_hash(optimizer),
        "checkpoint_sha256": sha256_bytes(
            fixed_npz_bytes(checkpoint_from_model(model))
        ),
    }


def train_fixed_manifest(
    manifest_path: Path, device: torch.device, label: str
) -> dict[str, Any]:
    """Run precisely one saved batch plan, never deriving data membership/order."""
    manifest = verify_manifest(manifest_path)
    paths = {name: Path(path) for name, path in manifest["artifact_paths"].items()}
    workdir = manifest_path.parent
    controls = configure_determinism(device, int(manifest["seed"]))
    rows = (
        read_jsonl(Path(manifest.get("replay_path", "")))
        if manifest.get("replay_path")
        else None
    )
    if rows is None:
        raise ValueError("manifest must identify replay_path")
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    selected = [rows[int(index)] for index in source]
    x = np.asarray([row["state"] for row in selected], dtype=np.float32)
    p = np.asarray([row["policy"] for row in selected], dtype=np.float32)
    v = np.asarray([row["value"] for row in selected], dtype=np.float32).reshape(-1, 1)
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    load_checkpoint_into_model(model, paths["initialization_checkpoint"])
    trainable = freeze_joint_heads(model)
    model.to(device).train()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=PR155_LR
    )
    tensors = [
        torch.from_numpy(array).to(device)
        for array in (x, p, v, legal_mask_matrix_for_encoded_states(x))
    ]
    captures: dict[str, Any] = {"initialization": _capture(model, optimizer)}
    policy_loss = torch.zeros((), device=device)
    value_loss = torch.zeros((), device=device)
    for step, batch in enumerate(plan, 1):
        indexes = torch.as_tensor(batch[batch >= 0], device=device, dtype=torch.long)
        logits, prediction = model(tensors[0][indexes])
        policy_loss = compute_policy_cross_entropy(
            logits.masked_fill(tensors[3][indexes] <= 0, -1e9), tensors[1][indexes]
        ).mean()
        value_loss = compute_value_loss_vector(
            prediction, tensors[2][indexes], value_loss="huber", huber_delta=1.0
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + PR155_VALUE_LOSS_WEIGHT * value_loss).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            ).item()
        )
        optimizer.step()
        if step in CAPTURE_STEPS or step % 100 == 0 or step == len(plan):
            captures[f"batch_{step}"] = {
                **_capture(model, optimizer),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "gradient_norm": gradient_norm,
            }
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
        value_loss=float(value_loss.item()),
    )
    validation = np.load(paths["validation_source_indexes"], allow_pickle=False)[:2048]
    validation_x = torch.from_numpy(
        np.asarray([rows[int(i)]["state"] for i in validation], dtype=np.float32)
    ).to(device)
    with torch.no_grad():
        logits, values = model(validation_x)
    return {
        "label": label,
        "controls": controls,
        "captures": captures,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "artifact": str(artifact),
        "artifact_weights_sha256": sha256_file(artifact / "weights.json"),
        "parameter_sha256": _hash_state(model),
        "validation_logits": logits.cpu().numpy(),
        "validation_values": values.cpu().numpy(),
        "trainable_parameter_names": trainable,
    }


def compare_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Apply the required exact-hash-or-output-tolerance reproducibility rule."""
    prediction_delta = max(
        float(np.max(np.abs(first["validation_logits"] - second["validation_logits"]))),
        float(np.max(np.abs(first["validation_values"] - second["validation_values"]))),
    )
    top1 = float(
        np.mean(
            np.argmax(first["validation_logits"], 1)
            == np.argmax(second["validation_logits"], 1)
        )
    )
    signs = float(
        np.mean((first["validation_values"] > 0) == (second["validation_values"] > 0))
    )
    exact = first["parameter_sha256"] == second["parameter_sha256"]
    capture_comparison = {
        label: {
            "parameters_identical": first["captures"][label]["parameter_sha256"]
            == second["captures"][label]["parameter_sha256"],
            "optimizer_identical": first["captures"][label]["optimizer_state_sha256"]
            == second["captures"][label]["optimizer_state_sha256"],
            "checkpoint_identical": first["captures"][label]["checkpoint_sha256"]
            == second["captures"][label]["checkpoint_sha256"],
        }
        for label in first["captures"]
    }
    return {
        "exact_parameter_hashes": exact,
        "checkpoint_hashes_identical": first["checkpoint_sha256"]
        == second["checkpoint_sha256"],
        "exported_weight_hashes_identical": first["artifact_weights_sha256"]
        == second["artifact_weights_sha256"],
        "maximum_prediction_difference": prediction_delta,
        "policy_top1_agreement": top1,
        "value_sign_agreement": signs,
        "capture_comparison": capture_comparison,
        "passes": exact or (prediction_delta <= 1e-7 and top1 == 1.0 and signs == 1.0),
    }


def _phase(row: dict[str, Any]) -> str:
    """Return the persisted phase or a deterministic game-progress fallback."""
    if row.get("phase") is not None:
        return str(row["phase"])
    ply = int(row.get("ply", row.get("move_index", 0)))
    length = max(1, int(row.get("game_length", 1)))
    return ("opening", "midgame", "late")[min(2, 3 * ply // length)]


def _state_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(row["state"], separators=(",", ":")).encode())


def select_search_probe_indexes(
    rows: list[dict[str, Any]],
    source_indexes: np.ndarray,
    seed: int,
    minimum: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic, capped round-robin sample across replay strata."""
    lengths = np.asarray([int(row.get("game_length", 0)) for row in rows])
    edges = np.quantile(lengths, np.linspace(0, 1, 11))
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in source_indexes.tolist():
        row = rows[int(index)]
        decile = str(
            min(
                9,
                int(
                    np.searchsorted(edges, int(row.get("game_length", 0)), side="right")
                    - 1
                ),
            )
        )
        key = (
            str(row.get("seat_context", f"player_{row.get('player', 0)}")),
            _phase(row),
            str(int(np.sign(float(row["value"])))),
            decile,
            str(int(np.clip(np.floor((float(row["value"]) + 1) * 2), 0, 3))),
        )
        buckets[key].append(int(index))
    rng = random.Random(seed)
    queues = deque()
    for key in sorted(buckets):
        values = buckets[key]
        rng.shuffle(values)
        queues.append((key, deque(values)))
    selected: list[int] = []
    state_counts: Counter[str] = Counter()
    game_counts: Counter[int] = Counter()
    selected_strata: Counter[str] = Counter()
    # One copy of a state and four positions per game avoid replay concentration.
    while queues and len(selected) < minimum:
        key, values = queues.popleft()
        while values:
            index = values.popleft()
            row = rows[index]
            state_hash = _state_hash(row)
            game = int(row["game_index"])
            if state_counts[state_hash] < 1 and game_counts[game] < 4:
                selected.append(index)
                state_counts[state_hash] += 1
                game_counts[game] += 1
                selected_strata["|".join(key)] += 1
                break
        if values:
            queues.append((key, values))
    if len(selected) < minimum:
        raise RuntimeError(
            f"only {len(selected)} capped unique search probe states are available"
        )
    metadata = {
        "selection_seed": seed,
        "minimum_states": minimum,
        "state_repeat_cap": 1,
        "game_repeat_cap": 4,
        "selected_states": len(selected),
        "stratum_counts": dict(sorted(selected_strata.items())),
        "state_hashes": [_state_hash(rows[index]) for index in selected],
        "game_ids": [int(rows[index]["game_index"]) for index in selected],
    }
    return np.asarray(selected, dtype=np.int64), metadata


def persist_search_probe(
    workdir: Path, rows: list[dict[str, Any]], validation_indexes: np.ndarray, seed: int
) -> dict[str, Any]:
    """Persist the single probe identity shared by every search budget."""
    indexes, metadata = select_search_probe_indexes(rows, validation_indexes, seed)
    indexes_path = workdir / "search_probe_source_indexes.npy"
    manifest_path = workdir / "search_probe_manifest.json"
    np.save(indexes_path, indexes, allow_pickle=False)
    metadata["source_indexes_sha256"] = sha256_file(indexes_path)
    metadata["source_indexes_path"] = str(indexes_path)
    write_json(manifest_path, metadata)
    metadata["manifest_sha256"] = sha256_file(manifest_path)
    return metadata


def value_metrics(
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    include_groups: bool = True,
) -> dict[str, Any]:
    """Measure terminal-outcome value quality, including auditable covariates."""
    targets = np.asarray([float(row["value"]) for row in rows], dtype=np.float64)
    predictions = np.asarray(
        [float(output["value"]) for output in outputs], dtype=np.float64
    )
    if not len(rows):
        return {
            "mae": 0.0,
            "sign_accuracy": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "calibration_buckets": [],
        }
    pearson = (
        float(np.corrcoef(targets, predictions)[0, 1])
        if np.std(targets) and np.std(predictions)
        else 0.0
    )

    def ranks(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values))

    spearman = (
        float(np.corrcoef(ranks(targets), ranks(predictions))[0, 1])
        if len(rows) > 1
        else 0.0
    )
    calibration = []
    for low, high in zip(np.linspace(-1, 1, 11)[:-1], np.linspace(-1, 1, 11)[1:]):
        mask = (predictions >= low) & (
            predictions < high if high < 1 else predictions <= high
        )
        calibration.append(
            {
                "low": float(low),
                "high": float(high),
                "count": int(mask.sum()),
                "prediction_mean": float(predictions[mask].mean())
                if mask.any()
                else None,
                "outcome_mean": float(targets[mask].mean()) if mask.any() else None,
            }
        )
    if not include_groups:
        return {
            "mae": float(np.abs(predictions - targets).mean()),
            "sign_accuracy": float(np.mean((predictions > 0) == (targets > 0))),
            "pearson": pearson,
            "spearman": spearman,
            "calibration_buckets": calibration,
        }
    groups: dict[str, dict[str, Any]] = {}
    game_lengths = np.asarray([int(row.get("game_length", 0)) for row in rows])
    edges = np.quantile(game_lengths, np.linspace(0, 1, 11))
    dimensions = {
        "player": [
            str(row.get("seat_context", f"player_{row.get('player', 0)}"))
            for row in rows
        ],
        "phase": [_phase(row) for row in rows],
        "outcome": [str(int(np.sign(float(row["value"])))) for row in rows],
        "game_length_decile": [
            str(min(9, int(np.searchsorted(edges, length, side="right") - 1)))
            for length in game_lengths
        ],
    }
    dimensions["player_x_outcome"] = [
        f"{player}|{outcome}"
        for player, outcome in zip(dimensions["player"], dimensions["outcome"])
    ]
    dimensions["phase_x_outcome"] = [
        f"{phase}|{outcome}"
        for phase, outcome in zip(dimensions["phase"], dimensions["outcome"])
    ]
    for field, values in dimensions.items():
        for key in sorted(set(values)):
            indexes = [index for index, value in enumerate(values) if value == key]
            groups[f"{field}:{key}"] = (
                value_metrics(
                    [rows[index] for index in indexes],
                    [outputs[index] for index in indexes],
                    include_groups=False,
                )
                if len(indexes) < len(rows)
                else {}
            )
    return {
        "mae": float(np.abs(predictions - targets).mean()),
        "sign_accuracy": float(np.mean((predictions > 0) == (targets > 0))),
        "pearson": pearson,
        "spearman": spearman,
        "calibration_buckets": calibration,
        "by_stratum": groups,
    }


def value_change_metrics(
    rows: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report candidate-minus-current value movement on every required stratum."""
    candidate_values = np.asarray([float(output["value"]) for output in candidate])
    current_values = np.asarray([float(output["value"]) for output in current])
    game_lengths = np.asarray([int(row.get("game_length", 0)) for row in rows])
    edges = np.quantile(game_lengths, np.linspace(0, 1, 11))
    dimensions = {
        "player": [
            str(row.get("seat_context", f"player_{row.get('player', 0)}"))
            for row in rows
        ],
        "phase": [_phase(row) for row in rows],
        "outcome": [str(int(np.sign(float(row["value"])))) for row in rows],
        "game_length_decile": [
            str(min(9, int(np.searchsorted(edges, length, side="right") - 1)))
            for length in game_lengths
        ],
    }
    dimensions["player_x_outcome"] = [
        f"{player}|{outcome}"
        for player, outcome in zip(dimensions["player"], dimensions["outcome"])
    ]
    dimensions["phase_x_outcome"] = [
        f"{phase}|{outcome}"
        for phase, outcome in zip(dimensions["phase"], dimensions["outcome"])
    ]

    def metrics(indexes: list[int]) -> dict[str, float | int]:
        return {
            "rows": len(indexes),
            "candidate_minus_current_value_delta": float(
                (candidate_values[indexes] - current_values[indexes]).mean()
            ),
            "value_sign_change_rate": float(
                np.mean(
                    (candidate_values[indexes] > 0) != (current_values[indexes] > 0)
                )
            ),
        }

    by_stratum = {}
    for field, values in dimensions.items():
        for key in sorted(set(values)):
            by_stratum[f"{field}:{key}"] = metrics(
                [i for i, value in enumerate(values) if value == key]
            )
    return {"global": metrics(list(range(len(rows)))), "by_stratum": by_stratum}


def suite_evaluation(
    *,
    workdir: Path,
    suite: Path,
    current: Path,
    candidate: Path,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    """Run both references on exactly one opening suite and report DS deltas."""
    report = run_opening_suite_benchmark(
        workdir=str(workdir),
        suite=str(suite),
        current=str(current),
        candidates=f"{current},{candidate}",
        budget_pairs="384:256,768:256,768:768,1200:1200,1200:256,256:768",
        games_per_opening=2,
        seed=seed,
        workers=workers,
        timeout=14400,
    )
    current_report = find_candidate_report(report, current.name)
    candidate_report = find_candidate_report(report, candidate.name)
    if current_report is None or candidate_report is None:
        raise RuntimeError("opening benchmark did not report both references")
    baseline, challenger = (
        benchmark_budget_results(current_report),
        benchmark_budget_results(candidate_report),
    )
    return {
        budget: {
            "raw_ds": challenger[budget]["ds"],
            "candidate_minus_current_ds": float(challenger[budget]["ds"])
            - float(baseline[budget]["ds"]),
            "current_minus_candidate_ds": float(baseline[budget]["ds"])
            - float(challenger[budget]["ds"]),
            **challenger[budget],
        }
        for budget in challenger
    }


def probe_gate(
    policy: dict[str, Any],
    current_policy: dict[str, Any],
    values: dict[str, Any],
    current_values: dict[str, Any],
    search: dict[str, Any],
    freeze_audit: dict[str, Any],
) -> bool:
    """Apply the prespecified replay and shared-search probe limits."""
    per_budget = search["changed_move_rate_by_budget_context"]
    return bool(
        policy["legal_failures"] == 0
        and freeze_audit["passes"]
        and policy["policy_kl"] < current_policy["policy_kl"]
        and policy["teacher_puct_top1_agreement"]
        > current_policy["teacher_puct_top1_agreement"]
        and values["mae"] < current_values["mae"]
        and values["sign_accuracy"] >= current_values["sign_accuracy"]
        and policy["changed_raw_top1_rate_vs_current"] <= 0.05
        and per_budget["768:768"]["changed_rate_vs_current"] <= 0.08
        and per_budget["1200:1200"]["changed_rate_vs_current"] <= 0.08
        and per_budget["1200:256"]["changed_rate_vs_current"] <= 0.10
    )


def medium_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        results["384:256"]["candidate_minus_current_ds"] >= 0.03
        and all(
            results[key]["candidate_minus_current_ds"] >= -0.03
            for key in ("768:768", "1200:1200", "1200:256")
        )
    )


def fixed_large_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        results["384:256"]["candidate_minus_current_ds"] >= 0.05
        and results["768:768"]["candidate_minus_current_ds"] >= -0.05
        and all(
            results[key]["candidate_minus_current_ds"] >= -0.03
            for key in ("1200:1200", "1200:256")
        )
    )


def classify_completed_evidence(stages: dict[str, bool | None]) -> str:
    """Prevent a candidate label when required evidence was skipped or missing."""
    if stages.get("reproducibility") is False:
        return "full_scale_training_nondeterministic"
    if stages.get("reproducibility") is not True or stages.get("probes") is None:
        return "deterministic_joint_heads_experiment_incomplete"
    if stages["probes"] is False:
        return "deterministic_joint_outcome_update_rejected"
    for stage in ("medium", "fixed_large", "heldout", "deterministic_gate"):
        if stages.get(stage) is None:
            return "deterministic_joint_heads_experiment_incomplete"
    if stages["heldout"] and stages["deterministic_gate"]:
        return "deterministic_joint_heads_candidate"
    return "joint_heads_safe_but_strength_neutral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--replay-audit", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def markdown(summary: dict[str, Any]) -> str:
    """Render a compact committed record; complete detail remains in JSON."""
    return "\n".join(
        [
            "# AlphaZero-Lite Deterministic Joint Heads Iteration Results",
            "",
            f"- classification: `{summary['classification']}`",
            f"- stop reasons: `{', '.join(summary['stop_reasons']) or 'none'}`",
            f"- batch-plan SHA256: `{summary['manifest']['batch_plan_sha256']}`",
            f"- reproducible: `{summary['reproducibility']['passes']}`",
            f"- freeze audit: `{summary.get('freeze_audit')}`",
            "",
            "## Complete Record",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current, replay, audit = (
        Path(args.current),
        Path(args.replay),
        Path(args.replay_audit),
    )
    if sha256_file(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if args.epochs != 1:
        raise ValueError("this protocol permits exactly one epoch")
    rows = read_jsonl(replay)
    manifest = build_manifest(
        rows=rows,
        workdir=workdir,
        current=current,
        replay=replay,
        replay_audit=audit,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    manifest_path = workdir / "training_manifest.json"
    manifest = verify_manifest(manifest_path)
    search_probe_manifest = persist_search_probe(
        workdir,
        rows,
        np.load(
            Path(manifest["artifact_paths"]["validation_source_indexes"]),
            allow_pickle=False,
        ),
        args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_a = train_fixed_manifest(
        manifest_path, device, "deterministic_joint_heads_e1_run_a"
    )
    run_b = train_fixed_manifest(
        manifest_path, device, "deterministic_joint_heads_e1_run_b"
    )
    reproducibility = compare_runs(run_a, run_b)
    stop_reasons = []
    if not reproducibility["passes"]:
        stop_reasons.append("same_device_training_diverged")
    freeze_audit = changed_key_audit(
        reference_checkpoint=workdir / "initialization.npz",
        candidate_checkpoint=Path(run_a["checkpoint"]),
        allowed_families={"policy", "value"},
    )
    if not freeze_audit["passes"]:
        stop_reasons.append("trunk_changed")
    summary = {
        "schema": "azlite_deterministic_joint_heads_iteration_v1",
        "manifest": manifest,
        "search_probe_manifest": search_probe_manifest,
        "device": str(device),
        "run_a": {k: v for k, v in run_a.items() if not isinstance(v, np.ndarray)},
        "run_b": {k: v for k, v in run_b.items() if not isinstance(v, np.ndarray)},
        "reproducibility": reproducibility,
        "freeze_audit": freeze_audit,
        "stop_reasons": stop_reasons,
    }
    if stop_reasons:
        summary["classification"] = (
            "full_scale_training_nondeterministic"
            if "same_device_training_diverged" in stop_reasons
            else "deterministic_joint_outcome_update_rejected"
        )
    else:
        # Scientific probes intentionally run only after the reproducibility gate.
        validation_indexes = np.load(
            Path(manifest["artifact_paths"]["validation_source_indexes"]),
            allow_pickle=False,
        )
        validation = [rows[int(i)] for i in validation_indexes]
        policy_rows = build_policy_probe_rows(validation)
        candidate = evaluate_raw_outputs(
            artifact_path=Path(run_a["artifact"]), rows=policy_rows
        )["outputs"]
        incumbent = evaluate_raw_outputs(artifact_path=current, rows=policy_rows)[
            "outputs"
        ]
        policy = evaluate_policy_probe_metrics(
            rows=policy_rows, candidate_outputs=candidate, current_outputs=incumbent
        )
        current_policy = evaluate_policy_probe_metrics(
            rows=policy_rows, candidate_outputs=incumbent, current_outputs=incumbent
        )
        values = value_metrics(validation, candidate)
        current_values = value_metrics(validation, incumbent)
        value_changes = value_change_metrics(validation, candidate, incumbent)
        search_indexes = np.load(
            workdir / "search_probe_source_indexes.npy", allow_pickle=False
        )
        search_rows = build_search_probe_rows(
            [rows[int(index)] for index in search_indexes],
            ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768"),
        )
        schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
        candidate_search = evaluate_search_outputs(
            artifact_path=Path(run_a["artifact"]),
            rows=search_rows,
            default_c_puct=args.default_c_puct,
            cpuct_schedule=schedule,
            tactical_root_bias=args.tactical_root_bias,
            seed=args.seed,
            workers=args.workers,
        )
        incumbent_search = evaluate_search_outputs(
            artifact_path=current,
            rows=search_rows,
            default_c_puct=args.default_c_puct,
            cpuct_schedule=schedule,
            tactical_root_bias=args.tactical_root_bias,
            seed=args.seed,
            workers=args.workers,
        )
        search = evaluate_search_probe_metrics(
            rows=search_rows,
            candidate_outputs=candidate_search,
            current_outputs=incumbent_search,
        )
        summary["policy_probe"] = {"candidate": policy, "current": current_policy}
        summary["value_probe"] = {
            "candidate": values,
            "current": current_values,
            "candidate_minus_current": value_changes,
        }
        summary["search_probe"] = search
        probe_pass = probe_gate(
            policy, current_policy, values, current_values, search, freeze_audit
        )
        if not probe_pass:
            stop_reasons.append("heldout_replay_probe_gate_failed")
            summary["classification"] = "deterministic_joint_outcome_update_rejected"
        else:
            medium = suite_evaluation(
                workdir=workdir / "medium",
                suite=Path(args.medium_suite),
                current=current,
                candidate=Path(run_a["artifact"]),
                seed=args.seed,
                workers=args.workers,
            )
            summary["medium"] = medium
            medium_pass = medium_gate(medium)
            if not medium_pass:
                stop_reasons.append("medium_strength_gate_failed")
                summary["classification"] = (
                    "joint_heads_probe_good_but_game_strength_bad"
                    if medium["384:256"]["candidate_minus_current_ds"] < 0
                    else "joint_heads_tradeoff"
                    if medium["384:256"]["candidate_minus_current_ds"] >= 0.03
                    else "joint_heads_safe_but_strength_neutral"
                )
            else:
                fixed = suite_evaluation(
                    workdir=workdir / "fixed_large",
                    suite=Path(args.fixed_large_suite),
                    current=current,
                    candidate=Path(run_a["artifact"]),
                    seed=args.seed,
                    workers=args.workers,
                )
                summary["fixed_large"] = fixed
                fixed_pass = fixed_large_gate(fixed)
                if not fixed_pass:
                    stop_reasons.append("fixed_large_strength_gate_failed")
                    summary["classification"] = "joint_heads_safe_but_strength_neutral"
                else:
                    heldout = {
                        str(path): suite_evaluation(
                            workdir=workdir / f"heldout_{index}",
                            suite=Path(path),
                            current=current,
                            candidate=Path(run_a["artifact"]),
                            seed=args.seed,
                            workers=args.workers,
                        )
                        for index, path in enumerate(args.heldout_suites.split(","))
                    }
                    summary["heldout"] = heldout
                    bootstrap = {
                        budget: bootstrap_ci(
                            [
                                result[budget]["candidate_minus_current_ds"]
                                for result in heldout.values()
                            ],
                            seed=args.seed,
                        )
                        for budget in ("384:256", "768:768", "1200:1200", "1200:256")
                    }
                    summary["heldout_bootstrap_95"] = bootstrap
                    final_pass = (
                        bootstrap["384:256"]["mean"] >= 0.05
                        and bootstrap["384:256"]["lower"] > 0.01
                        and bootstrap["768:768"]["mean"] >= -0.05
                        and bootstrap["1200:1200"]["mean"] >= -0.03
                        and bootstrap["1200:256"]["mean"] >= -0.03
                    )
                    if final_pass:
                        gate_path = workdir / "deterministic_gate.json"
                        command = [
                            str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
                            "--candidate-path",
                            str(run_a["artifact"]),
                            "--current-path",
                            str(current),
                            "--out",
                            str(gate_path),
                            "--workdir",
                            str(workdir / "deterministic_gate"),
                            "--budget-pairs",
                            "384:256,768:768,1200:1200,1200:256",
                            "--seed",
                            str(args.seed),
                            "--workers",
                            str(args.workers),
                            "--c-puct",
                            str(args.default_c_puct),
                            "--c-puct-schedule-json",
                            args.cpuct_schedule,
                            "--root-policy-mode",
                            "deterministic",
                            "--root-temperature",
                            "0",
                            "--tactical-root-bias",
                            str(args.tactical_root_bias),
                        ]
                        completed = subprocess.run(
                            command, cwd=REPO_ROOT, text=True, capture_output=True
                        )
                        if completed.returncode:
                            raise RuntimeError(
                                f"deterministic gate failed: {completed.stderr[-2000:]}"
                            )
                        gate = json.loads(gate_path.read_text(encoding="utf-8"))
                        summary["deterministic_gate"] = gate
                        gate_pass = gate.get("classification") in {
                            "standard_budget_breakthrough",
                            "high_search_breakthrough",
                        }
                        if gate_pass:
                            summary["classification"] = (
                                "deterministic_joint_heads_candidate"
                            )
                            stop_reasons.append("promotion_not_run_by_protocol")
                        else:
                            stop_reasons.append("deterministic_gate_failed")
                            summary["classification"] = (
                                "joint_heads_safe_but_strength_neutral"
                            )
                    else:
                        stop_reasons.append("heldout_strength_gate_failed")
                        summary["classification"] = (
                            "joint_heads_safe_but_strength_neutral"
                        )
        summary["stop_reasons"] = stop_reasons
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-deterministic-joint-heads-iteration-summary.json",
        summary,
    )
    (
        REPO_ROOT / "docs/alphazero-lite-deterministic-joint-heads-iteration-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "reproducible": reproducibility["passes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
