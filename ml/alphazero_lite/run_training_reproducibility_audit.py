#!/usr/bin/env python3
"""Reproducibility and provenance audit for the terminal-outcome value update.

Unlike :mod:`train`, this program never derives a split or permutation at run
time.  Both are immutable fixture inputs, so a changed result is attributable
to the numerical execution rather than data selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl,
)

MODEL_TYPE = "residual_v3"
INPUT_ENCODING = "kalah_v3"
HIDDEN_SIZES = (96, 3)
FIXTURE_TRAIN_ROWS = 2048
FIXTURE_VALIDATION_ROWS = 512
FIXTURE_BATCH_SIZE = 32
CAPTURE_STEPS = frozenset({1, 10, 50})
PR155_CONFIG = {
    "optimizer": {"type": "Adam", "lr": 5e-5, "weight_decay": 0.0},
    "batch_size": 512,
    "epochs": 2,
    "value_loss": "huber",
    "huber_delta": 1.0,
    "value_loss_weight": 1.25,
    "gradient_clip": 1.0,
    "lr_scheduler": "none",
    "model_mode": "train",
    "dtype": "float32",
    "freeze_map": {
        "trainable": [
            "value_hidden_layer.weight",
            "value_hidden_layer.bias",
            "value_head.weight",
            "value_head.bias",
        ],
        "frozen": "all other parameters",
    },
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 of bytes."""
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows in source order."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write canonical human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL with stable key and whitespace ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def fixed_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize arrays into a timestamp-free, sorted NPZ payload."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_output = io.BytesIO()
            np.lib.format.write_array(
                array_output, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_output.getvalue())
    return output.getvalue()


def write_fixed_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write deterministic NPZ serialization and return its SHA256."""
    payload = fixed_npz_bytes(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def parameter_hashes(model: PolicyValueNet) -> dict[str, str]:
    """Hash every state tensor by its exact CPU byte representation."""
    return {
        name: sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())
        for name, tensor in model.state_dict().items()
    }


def optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    """Hash optimizer tensors and scalar state without pickle serialization."""
    payload: list[tuple[str, str, str]] = []
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        for name, value in sorted(optimizer.state.get(parameter, {}).items()):
            if isinstance(value, torch.Tensor):
                encoded = value.detach().cpu().contiguous().numpy().tobytes().hex()
            else:
                encoded = repr(value)
            payload.append((str(index), str(name), encoded))
    return sha256_bytes(json.dumps(payload, separators=(",", ":")).encode())


def freeze_value_head(model: PolicyValueNet) -> None:
    """Freeze policy and trunk, leaving only the residual-v3 value head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert model.value_hidden_layer is not None
    for parameter in (
        *model.value_hidden_layer.parameters(),
        *model.value_head.parameters(),
    ):
        parameter.requires_grad = True


def configure_determinism(device: torch.device, seed: int) -> dict[str, Any]:
    """Set every supported deterministic control before creating the model."""
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
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def environment() -> dict[str, Any]:
    """Record hardware and framework information relevant to numerical results."""
    cuda = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu_model": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_available": cuda,
    }


def source_indexes_for_fixture(
    rows: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Select fixed source positions by ascending game id, preserving source order.

    The final selected game can be truncated to honor the exact row contract;
    the manifest records all source positions and game ids, removing ambiguity.
    """
    by_game: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_game[int(row["game_index"])].append(index)
    ordered = [index for game in sorted(by_game) for index in by_game[game]]
    if len(ordered) < FIXTURE_TRAIN_ROWS + FIXTURE_VALIDATION_ROWS:
        raise ValueError("replay is too small for the reproducibility fixture")
    return (
        ordered[:FIXTURE_TRAIN_ROWS],
        ordered[FIXTURE_TRAIN_ROWS : FIXTURE_TRAIN_ROWS + FIXTURE_VALIDATION_ROWS],
    )


def build_fixture(
    rows: list[dict[str, Any]], fixture_dir: Path, *, seed: int
) -> dict[str, Any]:
    """Create immutable train, validation, and two-epoch batch-plan files."""
    train_indexes, validation_indexes = source_indexes_for_fixture(rows)
    train_rows = [rows[index] for index in train_indexes]
    validation_rows = [rows[index] for index in validation_indexes]
    rng = np.random.default_rng(seed)
    plan = np.stack([rng.permutation(FIXTURE_TRAIN_ROWS) for _ in range(2)]).reshape(
        -1, FIXTURE_BATCH_SIZE
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)
    train_path = fixture_dir / "train.jsonl"
    validation_path = fixture_dir / "validation.jsonl"
    plan_path = fixture_dir / "batch_indexes.npy"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    np.save(plan_path, plan, allow_pickle=False)
    manifest = {
        "schema": "azlite_training_repro_fixture_v1",
        "seed": seed,
        "batch_size": FIXTURE_BATCH_SIZE,
        "epochs_in_plan": 2,
        "replay_index_vector_sha256": sha256_file(plan_path),
        "first_100_replay_indexes": plan.reshape(-1)[:100].tolist(),
        "last_100_replay_indexes": plan.reshape(-1)[-100:].tolist(),
        "train_source_row_indexes": train_indexes,
        "validation_source_row_indexes": validation_indexes,
        "train_game_ids": sorted({int(row["game_index"]) for row in train_rows}),
        "validation_game_ids": sorted(
            {int(row["game_index"]) for row in validation_rows}
        ),
        "files": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
            "batch_indexes.npy": sha256_file(plan_path),
        },
    }
    manifest["manifest_sha256_excluding_this_field"] = sha256_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    # A file cannot contain its literal SHA256 without changing that SHA.  The
    # manifest digest therefore covers its canonical payload excluding itself.
    write_json(fixture_dir / "manifest.json", manifest)
    return manifest


def capture(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    initial: dict[str, torch.Tensor],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Build a serialization-stable checkpoint record for one optimizer step."""
    arrays = checkpoint_from_model(model)
    deltas = {
        name: float(torch.max(torch.abs(value.detach().cpu() - initial[name])).item())
        for name, value in model.state_dict().items()
    }
    return {
        "checkpoint_sha256": sha256_bytes(fixed_npz_bytes(arrays)),
        "parameter_sha256": parameter_hashes(model),
        "optimizer_state_sha256": optimizer_hash(optimizer),
        "max_parameter_delta_from_initialization": max(deltas.values(), default=0.0),
        **metrics,
    }


def train_fixed_plan(
    *,
    init_checkpoint: Path,
    fixture_dir: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int = 1,
) -> dict[str, Any]:
    """Train a value-only model through immutable row-index batches."""
    controls = configure_determinism(device, seed)
    x, policy, value = load_jsonl(fixture_dir / "train.jsonl")
    plan = np.load(fixture_dir / "batch_indexes.npy", allow_pickle=False)
    expected_batches = epochs * (FIXTURE_TRAIN_ROWS // FIXTURE_BATCH_SIZE)
    if plan.shape != (
        2 * (FIXTURE_TRAIN_ROWS // FIXTURE_BATCH_SIZE),
        FIXTURE_BATCH_SIZE,
    ):
        raise ValueError("fixture batch plan has an unexpected shape")
    if epochs > 2:
        raise ValueError("fixture only persists two epochs")
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    load_checkpoint_into_model(model, init_checkpoint)
    freeze_value_head(model)
    model.to(device)
    model.train()
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=5e-5
    )
    x_all = torch.from_numpy(x).to(device)
    p_all = torch.from_numpy(policy).to(device)
    v_all = torch.from_numpy(value).to(device)
    masks = torch.from_numpy(legal_mask_matrix_for_encoded_states(x)).to(device)
    captures = {
        "initialization": capture(
            model,
            optimizer,
            initial,
            {"policy_loss": 0.0, "value_loss": 0.0, "gradient_norm": 0.0},
        )
    }
    for step, indexes in enumerate(plan[:expected_batches], start=1):
        index_tensor = torch.as_tensor(indexes, device=device, dtype=torch.long)
        logits, prediction = model(x_all[index_tensor])
        policy_loss = compute_policy_cross_entropy(
            logits.masked_fill(masks[index_tensor] <= 0, -1e9), p_all[index_tensor]
        ).mean()
        value_loss = compute_value_loss_vector(
            prediction, v_all[index_tensor], value_loss="huber", huber_delta=1.0
        ).mean()
        loss = policy_loss + (1.25 * value_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            ).item()
        )
        optimizer.step()
        if step in CAPTURE_STEPS or step == expected_batches:
            captures[f"batch_{step}"] = capture(
                model,
                optimizer,
                initial,
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "gradient_norm": grad_norm,
                },
            )
    checkpoint_path = output_dir / "checkpoint.npz"
    checkpoint_arrays = checkpoint_from_model(model)
    checkpoint_sha = write_fixed_npz(checkpoint_path, checkpoint_arrays)
    artifact_dir = output_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "weights.json"
    weights_path.write_text(
        json.dumps(
            {
                name: checkpoint_arrays[name].tolist()
                for name in sorted(checkpoint_arrays)
            }
        ),
        encoding="utf-8",
    )
    predictions = (
        model(
            torch.from_numpy(load_jsonl(fixture_dir / "validation.jsonl")[0]).to(device)
        )[1]
        .detach()
        .cpu()
        .numpy()
    )
    return {
        "controls": controls,
        "captures": captures,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "artifact_weights_sha256": sha256_file(weights_path),
        "prediction_sha256": sha256_bytes(predictions.tobytes()),
        "predictions": predictions,
    }


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare fixed-plan repeats, reporting the earliest divergent capture."""
    reference = runs[0]
    first_divergence = None
    for label, record in reference["captures"].items():
        for other_index, other in enumerate(runs[1:], start=2):
            if (
                record["parameter_sha256"]
                != other["captures"][label]["parameter_sha256"]
            ):
                keys = sorted(
                    set(record["parameter_sha256"])
                    | set(other["captures"][label]["parameter_sha256"])
                )
                first_divergence = {
                    "capture": label,
                    "run": other_index,
                    "parameter": next(
                        key
                        for key in keys
                        if record["parameter_sha256"].get(key)
                        != other["captures"][label]["parameter_sha256"].get(key)
                    ),
                }
                break
        if first_divergence:
            break
    max_difference = max(
        (
            float(np.max(np.abs(reference["predictions"] - run["predictions"])))
            for run in runs[1:]
        ),
        default=0.0,
    )
    return {
        "exact_tensor_hashes": first_divergence is None,
        "identical_exportable_checkpoint_hashes": len(
            {run["checkpoint_sha256"] for run in runs}
        )
        == 1,
        "identical_exported_weights_hashes": len(
            {run["artifact_weights_sha256"] for run in runs}
        )
        == 1,
        "maximum_prediction_difference": max_difference,
        "value_sign_agreement": min(
            (
                float(
                    np.mean((reference["predictions"] > 0) == (run["predictions"] > 0))
                )
                for run in runs[1:]
            ),
            default=1.0,
        ),
        "first_divergence": first_divergence,
    }


def provenance(
    rows: list[dict[str, Any]], args: argparse.Namespace, init_checkpoint: Path
) -> dict[str, Any]:
    """Record known configuration and explicitly list unavailable historical fields."""
    train_indexes, validation_indexes = source_indexes_for_fixture(rows)
    return {
        "schema": "azlite_training_provenance_comparison_v1",
        "environment": environment(),
        "current": {
            "weights_sha256": sha256_file(Path(args.current) / "weights.json"),
            "materialized_initialization_checkpoint_sha256": sha256_file(
                init_checkpoint
            ),
        },
        "pr155_candidate": {
            "weights_sha256": sha256_file(Path(args.pr155_candidate) / "weights.json")
        },
        "replay": {"sha256": sha256_file(Path(args.replay)), "rows": len(rows)},
        "fixture_selection": {
            "train_rows": len(train_indexes),
            "validation_rows": len(validation_indexes),
            "train_source_indexes": train_indexes[:100] + train_indexes[-100:],
            "validation_source_indexes": validation_indexes[:100]
            + validation_indexes[-100:],
        },
        "pr155_reconstructed_config": {
            **PR155_CONFIG,
            "seed": args.seed,
            "device": "cuda if available",
            "data_loader": "none; in-memory tensors",
            "batch_shuffle": "torch.randperm each epoch",
            "validation_split": "np.random.permutation unique source rows at 0.1",
            "checkpoint_serialization_order": "checkpoint_from_model insertion order",
            "artifact_export": "export_artifact.py JSON default serialization",
        },
        "pr161_config": {
            **PR155_CONFIG,
            "sampler": "row_uniform",
            "replay_index_generation": "random.Random(seed).shuffle over 85% game-prefix rows repeated per epoch",
        },
        "pr155_unreconstructible_fields": [
            "persisted train and validation source-row membership",
            "persisted replay-index vector and per-epoch torch permutations",
            "seed invocation immediately before model construction and training",
            "device, GPU model, CUDA, cuDNN, TF32, CUBLAS_WORKSPACE_CONFIG, deterministic-algorithm settings",
            "optimizer state and step-level checkpoints",
            "exact initialization checkpoint byte serialization",
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render the committed concise audit result."""
    cpu = summary["cpu"]
    gpu = summary["gpu"]
    lines = [
        "# AlphaZero-Lite Training Reproducibility Audit Results",
        "",
        f"- classification: `{summary['classification']}`",
        f"- current weights SHA256: `{summary['provenance']['current']['weights_sha256']}`",
        f"- PR #155 candidate weights SHA256: `{summary['provenance']['pr155_candidate']['weights_sha256']}`",
        "",
        "## Immutable Fixture",
        "",
        *[
            f"- {name}: `{digest}`"
            for name, digest in summary["fixture"]["files"].items()
        ],
        "",
        "## Repeated Runs",
        "",
        f"- CPU exact tensors: `{cpu['exact_tensor_hashes']}`; max prediction difference: `{cpu['maximum_prediction_difference']:.3g}`",
        f"- GPU: `{gpu}`",
        f"- first divergence: `{summary['first_divergence']}`",
        "",
        "## PR #155 Reconstruction",
        "",
        "- Full reconstruction and medium strength comparison are intentionally not claimed: PR #155 did not persist the split, replay-order vector, initialization serialization, or execution controls needed for a valid exact reconstruction.",
        "- The fixed-plan protocol establishes a repeatable replacement experiment without training a new objective or candidate.",
        "",
        "## Complete Audit Record",
        "",
        "The machine-readable record below contains the complete provenance comparison, fixture membership and hashes, repeated step-level tensor hashes, export hashes, and reconstruction/strength status.",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr155-candidate", required=True)
    parser.add_argument("--expected-pr155-candidate-sha256", required=True)
    parser.add_argument("--pr161-workdir", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    current_weights = Path(args.current) / "weights.json"
    candidate_weights = Path(args.pr155_candidate) / "weights.json"
    if sha256_file(current_weights) != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if sha256_file(candidate_weights) != args.expected_pr155_candidate_sha256:
        raise RuntimeError("PR #155 candidate weights hash mismatch")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.replay))
    init_checkpoint = materialize_weights_json_checkpoint(
        weights_path=current_weights, out_path=workdir / "initialization.npz"
    )
    provenance_payload = provenance(rows, args, init_checkpoint)
    write_json(workdir / "provenance_comparison.json", provenance_payload)
    fixture = build_fixture(rows, workdir / "fixtures", seed=args.seed)
    cpu_runs = [
        train_fixed_plan(
            init_checkpoint=init_checkpoint,
            fixture_dir=workdir / "fixtures",
            output_dir=workdir / f"cpu_{index}",
            device=torch.device("cpu"),
            seed=args.seed,
        )
        for index in range(1, 4)
    ]
    cpu = compare_runs(cpu_runs)
    if torch.cuda.is_available():
        gpu_runs = [
            train_fixed_plan(
                init_checkpoint=init_checkpoint,
                fixture_dir=workdir / "fixtures",
                output_dir=workdir / f"gpu_{index}",
                device=torch.device("cuda"),
                seed=args.seed,
            )
            for index in range(1, 4)
        ]
        gpu: dict[str, Any] = compare_runs(gpu_runs)
    else:
        gpu = {"status": "skipped_cuda_unavailable"}
    first_divergence = (
        cpu["first_divergence"]
        if cpu["first_divergence"]
        else gpu.get("first_divergence")
    )
    write_json(
        workdir / "first_divergence.json",
        first_divergence
        or {"classification": "none", "reason": "fixed-plan repeated runs matched"},
    )
    classification = (
        "pr155_artifact_not_reconstructible"
        if cpu["exact_tensor_hashes"]
        else "training_pipeline_nondeterministic"
    )
    summary = {
        "schema": "azlite_training_reproducibility_audit_v1",
        "classification": classification,
        "provenance": provenance_payload,
        "fixture": fixture,
        "cpu": cpu,
        "gpu": gpu,
        "first_divergence": first_divergence,
        "full_pr155_reconstruction": {
            "status": "not_valid_without_unrecoverable_provenance"
        },
        "minimal_medium_reproduction": {
            "status": "not_run; no valid reconstructed candidate"
        },
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-training-reproducibility-audit-summary.json",
        summary,
    )
    (
        REPO_ROOT / "docs/alphazero-lite-training-reproducibility-audit-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {"classification": classification, "cpu_exact": cpu["exact_tensor_hashes"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
