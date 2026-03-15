#!/usr/bin/env python3
"""Export a trained checkpoint into Rails-compatible artifact layout."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.input_encodings import BASE_FEATURE_ORDER, KALAH_V3_EXTRA_FEATURE_ORDER, feature_count_for


SUPPORTED_MODEL_TYPES = ["mlp_v1", "mlp_deep", "residual_v2", "residual_v3"]
RESIDUAL_MODEL_TYPES = {"residual_v2", "residual_v3"}


def feature_order_for(input_encoding: str) -> list[str]:
    if input_encoding == "kalah_v3":
        return [*BASE_FEATURE_ORDER, *KALAH_V3_EXTRA_FEATURE_ORDER]

    return list(BASE_FEATURE_ORDER)


def checkpoint_feature_count(npz: np.lib.npyio.NpzFile) -> int:
    if "w_input" in npz:
        return int(npz["w_input"].shape[0])
    if "w_hidden_1" in npz:
        return int(npz["w_hidden_1"].shape[0])
    if "w1" in npz:
        return int(npz["w1"].shape[0])
    raise ValueError("checkpoint is missing an input layer weight matrix")


def validate_checkpoint_feature_count(npz: np.lib.npyio.NpzFile, *, input_encoding: str) -> None:
    expected = feature_count_for(input_encoding)
    actual = checkpoint_feature_count(npz)
    if actual != expected:
        raise ValueError(f"checkpoint feature_count must be {expected} for {input_encoding}, got {actual}")


def validate_residual_v3_checkpoint(npz: np.lib.npyio.NpzFile) -> None:
    required_keys = [
        "w_policy_hidden",
        "b_policy_hidden",
        "w_value_hidden",
        "b_value_hidden",
    ]
    missing_keys = [key for key in required_keys if key not in npz]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"residual_v3 checkpoint must include specialized head weights: {missing}")

    trunk_size = int(npz["w_input"].shape[1])
    policy_hidden_shape = npz["w_policy_hidden"].shape
    value_hidden_shape = npz["w_value_hidden"].shape
    policy_hidden_size = int(npz["b_policy_hidden"].shape[0])
    value_hidden_size = int(npz["b_value_hidden"].shape[0])

    if policy_hidden_shape != (trunk_size, policy_hidden_size):
        raise ValueError(
            "residual_v3 checkpoint w_policy_hidden must have shape "
            f"({trunk_size}, {policy_hidden_size}), got {policy_hidden_shape}"
        )
    if value_hidden_shape != (trunk_size, value_hidden_size):
        raise ValueError(
            "residual_v3 checkpoint w_value_hidden must have shape "
            f"({trunk_size}, {value_hidden_size}), got {value_hidden_shape}"
        )
    if npz["w_policy"].shape != (policy_hidden_size, int(npz["b_policy"].shape[0])):
        raise ValueError(
            "residual_v3 checkpoint w_policy must have shape "
            f"({policy_hidden_size}, {int(npz['b_policy'].shape[0])}), got {npz['w_policy'].shape}"
        )
    if npz["w_value"].shape != (value_hidden_size, int(npz["b_value"].shape[0])):
        raise ValueError(
            "residual_v3 checkpoint w_value must have shape "
            f"({value_hidden_size}, {int(npz['b_value'].shape[0])}), got {npz['w_value'].shape}"
        )


def validate_residual_v2_checkpoint(npz: np.lib.npyio.NpzFile) -> None:
    specialized_head_keys = [
        "w_policy_hidden",
        "b_policy_hidden",
        "w_value_hidden",
        "b_value_hidden",
    ]
    present_keys = [key for key in specialized_head_keys if key in npz]
    if present_keys:
        present = ", ".join(present_keys)
        raise ValueError(
            "residual_v2 checkpoint cannot include residual_v3 specialized head weights: "
            f"{present}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Input .npz checkpoint")
    parser.add_argument("--out-dir", required=True, help="Output artifact directory")
    parser.add_argument("--version", required=True)
    parser.add_argument("--self-play-games", type=int, default=0)
    parser.add_argument("--policy-loss", type=float, default=0.0)
    parser.add_argument("--value-loss", type=float, default=0.0)
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default="mlp_v1")
    parser.add_argument("--rules-version", default="kalah_v1")
    parser.add_argument("--input-encoding", default="kalah_v1")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_model = out_dir / "model.npz"
    shutil.copy2(checkpoint, out_model)
    model_hash = sha256_file(out_model)

    npz = np.load(out_model)
    validate_checkpoint_feature_count(npz, input_encoding=args.input_encoding)
    weights_payload = {key: npz[key].tolist() for key in npz.files}

    architecture = {
        "type": "mlp_policy_value",
        "model_type": args.model_type,
        "activation": "relu",
        "policy_size": int(npz["w_policy"].shape[1]),
        "value_size": 1,
    }

    if args.model_type in RESIDUAL_MODEL_TYPES:
        trunk_size = int(npz["w_input"].shape[1])
        residual_block_count = 0
        while f"w_residual_{residual_block_count + 1}_1" in npz:
            residual_block_count += 1
        hidden_layer_count = 1 + (residual_block_count * 2)
        architecture_payload = {
            "type": "residual_policy_value",
            "hidden_sizes": [trunk_size],
            "hidden_layer_count": hidden_layer_count,
            "trunk_size": trunk_size,
            "residual_block_count": residual_block_count,
        }
        if args.model_type == "residual_v2":
            validate_residual_v2_checkpoint(npz)
        if args.model_type == "residual_v3":
            validate_residual_v3_checkpoint(npz)
            architecture_payload["hidden_layer_count"] += 2
        if args.model_type == "residual_v3" and "w_policy_hidden" in npz:
            architecture_payload["policy_hidden_size"] = int(npz["w_policy_hidden"].shape[1])
        if args.model_type == "residual_v3" and "w_value_hidden" in npz:
            architecture_payload["value_hidden_size"] = int(npz["w_value_hidden"].shape[1])
        architecture.update(architecture_payload)
    else:
        hidden_sizes: list[int] = []
        hidden_index = 1
        while f"w_hidden_{hidden_index}" in npz:
            hidden_sizes.append(int(npz[f"w_hidden_{hidden_index}"].shape[1]))
            hidden_index += 1

        if not hidden_sizes:
            hidden_sizes = [int(npz["w1"].shape[1]), int(npz["w2"].shape[1])]

        architecture.update(
            {
                "hidden_sizes": hidden_sizes,
                "hidden_layer_count": len(hidden_sizes),
            }
        )

    policy_size = int(npz["w_policy"].shape[1])
    weights_json_path = out_dir / "weights.json"
    weights_json_path.write_text(json.dumps(weights_payload), encoding="utf-8")
    weights_json_hash = sha256_file(weights_json_path)

    metadata = {
        "schema_version": "azlite_model_v1",
        "version": args.version,
        "game": "kalah",
        "rules_version": args.rules_version,
        "input_encoding": args.input_encoding,
        "feature_count": feature_count_for(args.input_encoding),
        "policy_size": policy_size,
        "feature_order": feature_order_for(args.input_encoding),
        "architecture": architecture,
        "normalization": {
            "pits_divisor": 48.0,
            "stores_divisor": 48.0,
            "current_player_encoding": "binary_0_1",
        },
        "training": {
            "self_play_games": args.self_play_games,
        },
        "metrics": {
            "policy_loss": args.policy_loss,
            "value_loss": args.value_loss,
        },
        "artifacts": {
            "weights_file": "weights.json",
            "weights_fallback_file": "model.npz",
            "weights_sha256": model_hash,
            "weights_json_sha256": weights_json_hash,
        },
        "framework": "numpy",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"exported artifact to {out_dir}")


if __name__ == "__main__":
    main()
