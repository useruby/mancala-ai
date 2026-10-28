"""Execute the frozen seed542 shard plan without replacing evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed542-selfplay-4096"
CHECKPOINT = ROOT / ".tmp/seed500-current-checkpoint.npz"
PYTHON = "/home/alex/Mancala/ai/.tmp/venv/bin/python"
INVENTORY = OUT / "execution-inventory.jsonl"
COMMON = [
    "-m",
    "ml.alphazero_lite.self_play",
    "--games",
    "64",
    "--checkpoint",
    ".tmp/seed500-current-checkpoint.npz",
    "--runtime-search-policy-artifact",
    "model-artifact/current",
    "--input-encoding",
    "kalah_v3",
    "--player-mode",
    "puct",
    "--simulations",
    "128",
    "--c-puct",
    "1.25",
    "--temperature-threshold",
    "10",
    "--temperature",
    "1.0",
    "--temperature-late",
    "0.1",
    "--dirichlet-alpha",
    "0.3",
    "--dirichlet-epsilon",
    "0.25",
    "--schedule-progress-mode",
    "none",
    "--max-moves",
    "200",
    "--workers",
    "1",
    "--policy-target-mode",
    "default",
    "--exact-root-policy-target-mode",
    "selected_one_hot",
    "--policy-target-noise-mode",
    "denoised",
    "--value-target-mode",
    "default",
    "--start-state-mode",
    "standard_4x6",
    "--write-game-metadata",
]


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, Any]:
    import numpy
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cwd": str(ROOT),
        "pid": os.getpid(),
    }


def record(event: dict[str, Any]) -> None:
    with INVENTORY.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_for(seed: int) -> list[str]:
    path = OUT / f"shard-{seed}.jsonl"
    return [
        PYTHON,
        "-m",
        "ml.alphazero_lite.self_play",
        "--out",
        str(path.relative_to(ROOT)),
        *COMMON[2:],
        "--seed",
        str(seed),
    ]


def run_seed(seed: int) -> None:
    target = OUT / f"shard-{seed}.jsonl"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {target}")
    if (
        digest(CHECKPOINT)
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise RuntimeError("initializer checkpoint authentication failed")
    command = command_for(seed)
    started = time.monotonic()
    started_at = timestamp()
    common = {
        "seed": seed,
        "source": "training" if seed < 542048 else "development",
        "command": command,
        "command_text": " ".join(command),
        "started_at": started_at,
        "environment": environment(),
    }
    record({**common, "event": "started"})
    log = OUT / f"shard-{seed}.log"
    try:
        with log.open("xb") as stream:
            process = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        exit_status = process.returncode
    except BaseException as error:
        record(
            {
                **common,
                "event": "interrupted",
                "ended_at": timestamp(),
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(error),
                "output_exists": target.exists(),
                "output_bytes": target.stat().st_size if target.exists() else 0,
                "output_sha256": digest(target),
            }
        )
        raise
    record(
        {
            **common,
            "event": "finished",
            "ended_at": timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "exit_status": exit_status,
            "output_exists": target.exists(),
            "output_bytes": target.stat().st_size if target.exists() else 0,
            "output_sha256": digest(target),
            "log_sha256": digest(log),
        }
    )
    if exit_status:
        raise subprocess.CalledProcessError(exit_status, command)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in range(542000, 542064):
        target = OUT / f"shard-{seed}.jsonl"
        if target.exists():
            record(
                {
                    "event": "preserved_existing_output_skipped",
                    "seed": seed,
                    "output_exists": True,
                    "output_bytes": target.stat().st_size,
                    "output_sha256": digest(target),
                    "timestamp": timestamp(),
                }
            )
            continue
        run_seed(seed)


if __name__ == "__main__":
    main()
