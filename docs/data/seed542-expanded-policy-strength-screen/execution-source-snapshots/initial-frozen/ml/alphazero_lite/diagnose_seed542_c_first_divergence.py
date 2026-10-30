"""Compare production seed504 and fixed-schedule C one update at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import run_seed504, run_seed542_fixed_update_cd as cd, train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed542-fixed-update-cd-diagnostics/first-divergence"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = value
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "strides_bytes": list(array.strides),
        "contiguous": bool(array.flags.c_contiguous),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def optimizer_record(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    state = {}
    for name, parameter in parameters:
        item = optimizer.state.get(parameter, {})
        state[name] = {
            key: array_record(value)
            for key, value in item.items()
            if isinstance(value, torch.Tensor)
        } | {
            key: value
            for key, value in item.items()
            if not isinstance(value, torch.Tensor)
        }
    return {
        "parameter_order": [name for name, _parameter in parameters],
        "options": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "state": state,
    }


class Capture:
    def __init__(self, model: train.PolicyValueNet):
        self.model = model
        self.items: list[dict[str, Any]] = []
        self.forward: dict[str, Any] | None = None
        self.row_ids: list[str] = []
        self.arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.legal_masks: np.ndarray | None = None
        self.observed_losses: dict[str, float] | None = None
        self.clip_function_norm: float | None = None
        self.model.register_forward_hook(self.forward_hook)

    def forward_hook(self, _module: Any, args: tuple[Any, ...], result: Any) -> None:
        logits, value = result
        self.forward = {
            "input": array_record(args[0]),
            "logits": array_record(logits),
            "value": array_record(value),
            "model_training_mode": bool(self.model.training),
        }

    def batch_record(
        self, indexes: list[int], *, production_indexing: bool
    ) -> dict[str, Any]:
        if self.arrays is None:
            raise RuntimeError("diagnostic_batch_arrays_unbound")
        idx = np.asarray(indexes, dtype=np.int64)
        states, policies, values, weights = self.arrays
        if self.legal_masks is None:
            raise RuntimeError("diagnostic_legal_masks_unbound")
        if production_indexing:
            tensor_index = torch.from_numpy(idx)
            selected = [torch.from_numpy(array)[tensor_index] for array in self.arrays]
            all_legal = torch.from_numpy(self.legal_masks)
            legal_mask = all_legal[tensor_index]
        else:
            selected = [torch.from_numpy(array[idx]) for array in self.arrays]
            legal_mask = torch.from_numpy(self.legal_masks[idx])
        return {
            "row_ids": [self.row_ids[int(index)] for index in indexes],
            "batch_size": len(indexes),
            "model_training_mode": bool(self.model.training),
            "tensors": {
                "input": array_record(selected[0]),
                "target_policy": array_record(selected[1]),
                "target_value": array_record(selected[2]),
                "policy_weight": array_record(selected[3]),
                "legal_mask": array_record(legal_mask),
            },
        }

    def production_callback(self, phase: str, context: dict[str, Any]) -> None:
        if phase == "before":
            indexes = context["batch_indexes"]
            batch = self.batch_record(indexes, production_indexing=True)
            self.items.append(
                {
                    "row_ids": batch["row_ids"],
                    "batch_size": batch["batch_size"],
                    "forward": self.forward,
                    "batch": batch,
                    "raw_gradients": {
                        name: array_record(value)
                        for name, value in context["raw_gradients"].items()
                    },
                    "gradient_norm_preclip": context["gradient_norm"],
                    "optimizer_before": optimizer_record(
                        context["optimizer"], self.model
                    ),
                    "parameters_before": cd.state_digest(self.model),
                    "losses": {
                        "policy_loss": context["policy_loss"],
                        "value_loss": context["value_loss"],
                        **(self.observed_losses or {}),
                        "gradient_norm_preclip_manual": context["gradient_norm"],
                    },
                }
            )
        else:
            self.items[-1]["clipped_gradients"] = {
                name: array_record(parameter.grad)
                for name, parameter in self.model.named_parameters()
            }
            self.items[-1]["optimizer_after"] = optimizer_record(
                context["optimizer"], self.model
            )
            self.items[-1]["losses"]["gradient_norm_clip_function"] = (
                self.clip_function_norm
            )
            self.items[-1]["parameters_after"] = cd.state_digest(self.model)


def binding(output: Path) -> dict[str, Any]:
    manifest = json.loads((ROOT / "docs/data/seed504/manifest.json").read_text())
    protocol = ROOT / "docs/data/seed542-selfplay-4096/training-protocol.json"
    return {
        "status": "prospective diagnostic trace binding",
        "protocol_sha256": sha(protocol),
        "sources": {
            relative: sha(ROOT / relative)
            for relative in (
                "ml/alphazero_lite/train.py",
                "ml/alphazero_lite/run_seed504.py",
                "ml/alphazero_lite/run_seed542_fixed_update_cd.py",
                "ml/alphazero_lite/diagnose_seed542_c_first_divergence.py",
                "docs/data/seed504/manifest.json",
                "docs/data/seed504/permutations.json",
                "docs/data/seed542-selfplay-4096/training-protocol.json",
                "docs/data/seed542-selfplay-4096/initializer.npz",
            )
        },
        "training_sources": {
            source["path"]: sha(ROOT / source["path"])
            for source in manifest["sources"]["training"]
        },
        "historical_checkpoints": {
            f"E{epoch}": sha(ROOT / f"docs/data/seed504/seed504_epoch{epoch}.npz")
            for epoch in range(1, 5)
        },
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "BLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_config": torch.__config__.show(),
        },
        "command": sys.argv,
        "outer_command": os.environ.get("SEED542_CD_DIAGNOSTIC_LAUNCH_COMMAND"),
        "output": str(output),
    }


def production_trace() -> list[dict[str, Any]]:
    training, _development, _split_evidence = run_seed504.split()
    paths = [
        ROOT / f"docs/data/seed500-selfplay-512/shard-{i}.jsonl"
        for i in range(503000, 503006)
    ]
    x, p, v, replay, weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(x) != len(training) or not np.array_equal(replay, np.arange(len(training))):
        raise ValueError("diagnostic_seed504_loader_identity_mismatch")
    model = run_seed504.load_model(run_seed504.INITIALIZER, torch.device("cpu"))
    train.set_seed(504)
    capture = Capture(model)
    capture.row_ids = [
        f"docs/data/seed500-selfplay-512/{row['_source']}:{row['_line']}"
        for row in training
    ]
    capture.arrays = (x, p, v, weights)
    capture.legal_masks = train.legal_mask_matrix_for_encoded_states(x)
    production_train_one_epoch = train.train_one_epoch
    original_clip = torch.nn.utils.clip_grad_norm_

    def production_clip(
        parameters: Any, max_norm: float, *args: Any, **kwargs: Any
    ) -> Any:
        result = original_clip(parameters, max_norm, *args, **kwargs)
        capture.clip_function_norm = float(result.detach().cpu().item())
        return result

    def instrumented_train_one_epoch(**kwargs: Any) -> dict[str, Any]:
        kwargs["step_callback"] = capture.production_callback
        kwargs["step_callback_needs_raw_gradients"] = True
        kwargs["loss_observer"] = lambda context: setattr(
            capture,
            "observed_losses",
            {
                "weighted_total_objective": float(
                    context["total_loss_tensor"].detach().cpu().item()
                )
            },
        )
        return production_train_one_epoch(**kwargs)

    train.train_one_epoch = instrumented_train_one_epoch
    torch.nn.utils.clip_grad_norm_ = production_clip
    try:
        train.train(
            model,
            x,
            p,
            v,
            replay,
            policy_loss_weights=weights,
            epochs=4,
            batch_size=512,
            lr=0.0001,
            device=torch.device("cpu"),
            value_loss_weight=0.5,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.0,
            grad_clip=1.0,
            save_top_k=1,
            lr_scheduler="none",
            weight_decay=0.0,
            final_checkpoint="final",
            primary_order_seed=504,
        )
    finally:
        train.train_one_epoch = production_train_one_epoch
        torch.nn.utils.clip_grad_norm_ = original_clip
    return capture.items


def custom_trace() -> list[dict[str, Any]]:
    schedule = cd.load_schedule()
    x, p, v, weights, row_ids = cd.load_sources("C")
    positions = {row_id: i for i, row_id in enumerate(row_ids)}
    model = cd.load_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    capture = Capture(model)
    capture.row_ids = row_ids
    capture.arrays = (x, p, v, weights)
    capture.legal_masks = train.legal_mask_matrix_for_encoded_states(x)
    original_clip = torch.nn.utils.clip_grad_norm_
    items = []
    for step in schedule["updates"]:
        ids = step["C"]
        current: dict[str, Any] = {"row_ids": ids, "batch_size": len(ids)}
        batch_indexes = [positions[row_id] for row_id in ids]
        current["batch"] = capture.batch_record(
            batch_indexes, production_indexing=False
        )

        def clip_capture(
            parameters: Any, max_norm: float, *args: Any, **kwargs: Any
        ) -> Any:
            params = list(parameters)
            current["raw_gradients"] = {
                name: array_record(parameter.grad)
                for name, parameter in model.named_parameters()
            }
            current["gradient_norm_preclip"] = float(
                torch.sqrt(
                    sum(
                        torch.sum(parameter.grad.detach() ** 2)
                        for parameter in params
                        if parameter.grad is not None
                    )
                )
            )
            value = original_clip(params, max_norm, *args, **kwargs)
            current["gradient_norm_clip_function"] = float(value.detach().cpu().item())
            current["clipped_gradients"] = {
                name: array_record(parameter.grad)
                for name, parameter in model.named_parameters()
            }
            return value

        torch.nn.utils.clip_grad_norm_ = clip_capture
        try:
            current["optimizer_before"] = optimizer_record(optimizer, model)
            current["parameters_before"] = cd.state_digest(model)
            result = cd.update(model, optimizer, ids, positions, x, p, v, weights)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        current["forward"] = capture.forward
        current["optimizer_after"] = optimizer_record(optimizer, model)
        current["parameters_after"] = cd.state_digest(model)
        current["losses"] = {
            key: result[key]
            for key in (
                "policy_loss",
                "value_loss",
                "weighted_total_objective",
            )
        }
        current["losses"]["gradient_norm_preclip_manual"] = current[
            "gradient_norm_preclip"
        ]
        current["losses"]["gradient_norm_clip_function"] = current[
            "gradient_norm_clip_function"
        ]
        items.append(current)
    return items


def compare_arrays(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(compare_arrays(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            compare_arrays(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, bytes) and isinstance(b, bytes):
        return a == b
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"diagnostic_output_not_empty:{args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    record = binding(args.out)
    (args.out / "diagnostic-registration.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    production = production_trace()
    custom = custom_trace()
    if len(production) != len(custom):
        raise ValueError("diagnostic_update_count_mismatch")
    comparisons = []
    divergent_update = None
    first_step_detail = None
    for index, (left, right) in enumerate(zip(production, custom, strict=True), 1):
        fields = (
            "batch",
            "forward",
            "losses",
            "raw_gradients",
            "clipped_gradients",
            "optimizer_before",
            "optimizer_after",
            "parameters_before",
            "parameters_after",
        )
        status = {
            "update": index,
            "row_ids_equal": left.get("row_ids") == right.get("row_ids"),
            "batch_size_equal": left.get("batch_size") == right.get("batch_size"),
            "fields_equal": {
                field: compare_arrays(left.get(field), right.get(field))
                for field in fields
            },
        }
        comparisons.append(status)
        if divergent_update is None and (
            not status["row_ids_equal"]
            or not status["batch_size_equal"]
            or not all(status["fields_equal"].values())
        ):
            divergent_update = index
            first_step_detail = {"production": left, "corrected_C": right}
    report = {
        "updates_compared": len(comparisons),
        "first_divergent_update": divergent_update,
        "production_and_corrected_C_match_all_updates": divergent_update is None,
        "per_update": comparisons,
        "first_divergence_detail": first_step_detail,
    }
    (args.out / "first-divergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in report
                if key != "per_update" and key != "first_divergence_detail"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
