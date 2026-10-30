"""Bound diagnostics for the seed542 control reproduction discrepancy."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import run_seed504, train

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INITIALIZER_SHA256 = (
    "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding(output: Path) -> dict[str, Any]:
    manifest_path = ROOT / "docs/data/seed504/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {
        "docs/data/seed504/manifest.json": sha256(manifest_path),
        "docs/data/seed504/permutations.json": sha256(
            ROOT / "docs/data/seed504/permutations.json"
        ),
        "docs/data/seed504/split.json": sha256(ROOT / "docs/data/seed504/split.json"),
        "docs/data/seed542-selfplay-4096/initializer.npz": sha256(
            ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
        ),
        "initializer_at_seed504_path": sha256(run_seed504.INITIALIZER),
        **{
            source["path"]: sha256(ROOT / source["path"])
            for source in [
                *manifest["sources"]["training"],
                *manifest["sources"]["development"],
            ]
        },
        **{
            f"docs/data/seed504/seed504_epoch{epoch}.npz": sha256(
                ROOT / f"docs/data/seed504/seed504_epoch{epoch}.npz"
            )
            for epoch in range(1, 5)
        },
    }
    packages = sorted(
        (
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
            and distribution.metadata["Name"].lower() != "pip"
        ),
        key=str.casefold,
    )
    return {
        "status": "diagnostic reference prospective binding",
        "runner_sha256": sha256(Path(run_seed504.__file__)),
        "train_source_sha256": sha256(Path(train.__file__)),
        "diagnostic_harness_sha256": sha256(Path(__file__)),
        "input_sha256": inputs,
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "packages": packages,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "torch_openmp_threads": torch.get_num_threads(),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_config": torch.__config__.show(),
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
        },
        "command": sys.argv,
        "outer_command": os.environ.get("SEED542_CD_DIAGNOSTIC_LAUNCH_COMMAND"),
        "planned_production_runner_call": "run_seed504.run(out=diagnostic_output, do_train=True)",
        "historical_torch": json.loads(manifest_path.read_text())["torch"],
    }


def compare_checkpoints(actual_path: Path, historical_path: Path) -> dict[str, Any]:
    actual = np.load(actual_path)
    historical = np.load(historical_path)
    per_parameter = {}
    for name in sorted(set(actual.files) | set(historical.files)):
        if name not in actual or name not in historical:
            per_parameter[name] = {
                "present_actual": name in actual,
                "present_historical": name in historical,
            }
            continue
        left, right = actual[name], historical[name]
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        per_parameter[name] = {
            "shape_actual": list(left.shape),
            "shape_historical": list(right.shape),
            "dtype_actual": str(left.dtype),
            "dtype_historical": str(right.dtype),
            "bitwise_equal": bool(np.array_equal(left, right)),
            "differing_elements": int(np.count_nonzero(left != right)),
            "maximum_absolute_difference": float(delta.max(initial=0.0)),
        }
    return {
        "actual_sha256": sha256(actual_path),
        "historical_sha256": sha256(historical_path),
        "parameter_names_equal": set(actual.files) == set(historical.files),
        "bitwise_equal": all(
            item.get("bitwise_equal", False) for item in per_parameter.values()
        ),
        "per_parameter": per_parameter,
    }


def run(output: Path) -> dict[str, Any]:
    if sha256(run_seed504.INITIALIZER) != EXPECTED_INITIALIZER_SHA256:
        raise ValueError("diagnostic_initializer_sha256_mismatch")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic_output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    registration = source_binding(output)
    registration_path = output / "diagnostic-registration.json"
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    run_seed504.run(output, do_train=True)
    comparison = compare_checkpoints(
        output / "seed504_epoch1.npz",
        ROOT / "docs/data/seed504/seed504_epoch1.npz",
    )
    summary = {
        "production_runner_reproduces_historical_E1": comparison["bitwise_equal"],
        "historical_runtime_torch": registration["historical_torch"],
        "current_runtime_torch": torch.__version__,
        "E1_comparison": comparison,
    }
    (output / "production-reference-comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/seed542-fixed-update-cd-diagnostics/production-reference",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
