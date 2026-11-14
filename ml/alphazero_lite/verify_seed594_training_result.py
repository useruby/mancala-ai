"""Read-only verifier for a completed Seed 594 training treatment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import train
from ml.alphazero_lite.seed594_exact_oracle_distillation import (
    ARTIFACTS_ROOT,
    CONFIG_PATH,
    FREEZE_PATH,
    MODEL_HASHES,
    RUNTIME,
    sha256_bytes,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _fail(reason: str) -> None:
    raise ValueError(reason)


def verify_training_result(output: Path, *, root: Path) -> dict[str, Any]:
    """Validate a result directory without mutating it or running inference."""
    freeze = _json(root / FREEZE_PATH)
    config = _json(root / CONFIG_PATH)
    order = config["training"]["batch_order"]
    if len(order) != 256 or sorted(order) != list(range(256)):
        _fail("prescribed_batch_order_invalid")
    expected_batches = [order[i : i + 32] for i in range(0, 256, 32)]
    trace_path = output / "training-trace.jsonl"
    checkpoint_path = output / "seed594-exact-oracle-update-512.npz"
    identity_path = output / "checkpoint-identity.json"
    for path in (trace_path, checkpoint_path, identity_path):
        if not path.is_file():
            _fail(f"result_file_missing:{path.name}")

    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(trace) != 512:
        _fail("training_trace_update_count_mismatch")
    for index, row in enumerate(trace):
        if row.get("update") != index + 1:
            _fail("training_trace_update_identity_mismatch")
        if row.get("root_indexes") != expected_batches[index % 8]:
            _fail("training_trace_batch_identity_mismatch")
        if row.get("seed") != 571 or row.get("runtime") != RUNTIME:
            _fail("training_trace_runtime_identity_mismatch")
        if not np.isfinite(row.get("loss", np.nan)) or not np.isfinite(
            row.get("gradient_norm", np.nan)
        ):
            _fail("training_trace_nonfinite_value")

    identity = _json(identity_path)
    checkpoint_hash = sha256_bytes(checkpoint_path.read_bytes())
    target_identity = freeze["target_identity"]
    expected_identity = {
        "checkpoint": checkpoint_path.name,
        "sha256": checkpoint_hash,
        "target_sha256": target_identity["canonical_sha256"],
        "training_roots": 256,
        "updates": 512,
        "frozen_parameters_unchanged": True,
        "parent_policy_checkpoint_sha256": MODEL_HASHES["P-E4/model.npz"],
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        _fail("checkpoint_identity_binding_mismatch")

    parent_path = root / ARTIFACTS_ROOT.relative_to(root) / "P-E4/model.npz"
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, parent_path):
        _fail("initializer_checkpoint_tensor_mismatch")
    arrays = train.checkpoint_from_model(model)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        actual_arrays = {name: archive[name] for name in archive.files}
    if set(actual_arrays) != set(arrays):
        _fail("checkpoint_tensor_names_mismatch")
    tensor_identities: dict[str, dict[str, Any]] = {}
    for name, expected in arrays.items():
        value = actual_arrays[name]
        if value.shape != expected.shape:
            _fail(f"checkpoint_tensor_shape_mismatch:{name}")
        if not np.isfinite(value).all():
            _fail(f"checkpoint_tensor_nonfinite:{name}")
        tensor_identities[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if identity.get("tensors") != tensor_identities:
        _fail("checkpoint_tensor_identity_mismatch")

    # The value pathway is frozen by the registered trainable-prefix contract.
    for name, original in arrays.items():
        if not name.startswith(
            ("input_layer.", "residual_layers.", "policy_hidden_layer.", "policy_head.")
        ):
            if not np.array_equal(actual_arrays[name], original):
                _fail(f"frozen_parameter_changed:{name}")

    receipt = {
        "schema": "seed594-training-result-verification-v1",
        "status": "verified",
        "updates": len(trace),
        "batch_order_sha256": hashlib.sha256(
            json.dumps(order, separators=(",", ":")).encode()
        ).hexdigest(),
        "trace_sha256": sha256_bytes(trace_path.read_bytes()),
        "checkpoint_sha256": checkpoint_hash,
        "target_file_sha256": target_identity["file_sha256"],
        "target_canonical_sha256": target_identity["canonical_sha256"],
        "initializer_sha256": MODEL_HASHES["P-E4/model.npz"],
        "frozen_parameters_unchanged": True,
        "tensor_count": len(actual_arrays),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_training_result(args.result_dir, root=args.root), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
