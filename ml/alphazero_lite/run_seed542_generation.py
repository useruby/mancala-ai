"""Execute the frozen seed542 shard plan without replacing evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed542-selfplay-4096"
FREEZE_RELATIVE = Path("docs/data/seed542-selfplay-4096/continuation-freeze.json")
FREEZE_SHA256 = "0c70b483d4ca7417c92f2fa5da1f2c76913140bbc9cced8502cc320e79b01fb5"
PYTHON = "/home/alex/Mancala/ai/.tmp/venv/bin/python"
INVENTORY = OUT / "execution-inventory.jsonl"
CONTINUATION_INVENTORY = OUT / "continuation-inventory.jsonl"
COMMAND_TEMPLATE = [
    PYTHON,
    "-m",
    "ml.alphazero_lite.self_play",
    "--out",
    "{output}",
    "--games",
    "64",
    "--seed",
    "{seed}",
    "--checkpoint",
    "docs/data/seed542-selfplay-4096/initializer.npz",
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
SEEDS = range(542000, 542064)
SOURCE_PATHS = {
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/pipeline.py",
}
BEHAVIOR_PATHS = {
    "weights": "model-artifact/current/weights.json",
    "runtime_search_policy": "model-artifact/current/search_policy.json",
    "initializer": "docs/data/seed542-selfplay-4096/initializer.npz",
    "native_probe": "model-artifact/runtime/kalah_v1_tablebase",
    "tablebase": "model-artifact/runtime/kalah_v1_21.kvtb",
}
RESUME_POLICY = {
    "interrupted_or_ambiguous_attempts": "preserve and skip; never retry automatically",
    "completed_outputs": "verify inventory hash and skip",
    "only_unattempted_seeds_may_launch": True,
    "new_execution_records": "docs/data/seed542-selfplay-4096/continuation-inventory.jsonl",
    "generation_requires_explicit_execute_flag": True,
}
RESOURCE_SCOPE = {
    "cpu_quota": "25%",
    "cpu_weight": 1,
    "memory_max_bytes": 4294967296,
    "nice": 19,
    "cpu_affinity": "31",
}


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


def record(event: dict[str, Any], *, inventory: Path | None = None) -> None:
    inventory = inventory or INVENTORY
    with inventory.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_for(seed: int) -> list[str]:
    path = OUT / f"shard-{seed}.jsonl"
    return [
        token.format(output=path.relative_to(ROOT).as_posix(), seed=seed)
        for token in COMMAND_TEMPLATE
    ]


def preflight(
    *,
    root: Path = ROOT,
    freeze_path: Path | None = None,
    expected_freeze_sha256: str = FREEZE_SHA256,
    command_template: list[str] | None = None,
    current_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Reject source, runtime, dependency, or command drift before launch."""
    freeze_path = freeze_path or root / FREEZE_RELATIVE
    freeze_bytes = freeze_path.read_bytes()
    freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    if freeze_sha256 != expected_freeze_sha256:
        raise RuntimeError("generation_preflight_freeze_sha256_mismatch")
    frozen = json.loads(freeze_bytes)
    active_template = COMMAND_TEMPLATE if command_template is None else command_template
    if active_template != frozen["command_template"]:
        raise RuntimeError("generation_preflight_command_template_drift")
    if set(frozen["source_sha256"]) != SOURCE_PATHS:
        raise RuntimeError("generation_preflight_source_manifest_incomplete")
    if set(frozen["behavior"]) != set(BEHAVIOR_PATHS) | {"model_version"}:
        raise RuntimeError("generation_preflight_behavior_manifest_incomplete")
    if frozen["behavior"]["model_version"] != "seed48-nextgen-s455-default-value-iter1":
        raise RuntimeError("generation_preflight_behavior_model_drift")
    if any(
        frozen["behavior"][name].get("path") != path
        for name, path in BEHAVIOR_PATHS.items()
    ):
        raise RuntimeError("generation_preflight_behavior_path_drift")
    if frozen["resume_policy"] != RESUME_POLICY:
        raise RuntimeError("generation_preflight_resume_policy_drift")
    if frozen["resource_scope"] != RESOURCE_SCOPE:
        raise RuntimeError("generation_preflight_resource_scope_drift")
    if frozen["seed_plan"] != {
        "first_seed": 542000,
        "last_seed_inclusive": 542063,
        "games_per_shard": 64,
        "training_seed_count": 48,
        "training_last_seed_inclusive": 542047,
        "development_seed_count": 16,
        "development_first_seed": 542048,
    }:
        raise RuntimeError("generation_preflight_seed_plan_drift")
    for relative, expected in frozen["source_sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(
                f"generation_preflight_source_sha256_mismatch:{relative}"
            )
    for name in BEHAVIOR_PATHS:
        artifact = frozen["behavior"][name]
        path = root / artifact["path"]
        actual = digest(path)
        if actual != artifact["sha256"]:
            if path.suffix == ".kvtb" and path.is_file():
                pointer = re.search(rb"oid sha256:([0-9a-f]{64})\n", path.read_bytes())
                if pointer and pointer.group(1).decode() == artifact["sha256"]:
                    raise RuntimeError(
                        "generation_preflight_tablebase_not_materialized"
                    )
            raise RuntimeError(
                f"generation_preflight_artifact_sha256_mismatch:{artifact['path']}"
            )
    observed_environment = current_environment or {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
        "torch": __import__("torch").__version__,
    }
    if observed_environment != frozen["environment"]:
        raise RuntimeError("generation_preflight_environment_drift")
    executable = Path(active_template[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("generation_preflight_python_executable_unavailable")
    if Path(sys.executable).resolve() != executable.resolve():
        raise RuntimeError("generation_preflight_runner_interpreter_mismatch")
    return frozen


def inventory_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def seed_state(seed: int, *, root: Path | None = None) -> str:
    """Classify evidence without retrying completed or ambiguous attempts."""
    root = root or ROOT
    out = root / "docs/data/seed542-selfplay-4096"
    shard = out / f"shard-{seed}.jsonl"
    log = out / f"shard-{seed}.log"
    events = inventory_events(out / "execution-inventory.jsonl")
    events += inventory_events(out / "continuation-inventory.jsonl")
    seed_events = [event for event in events if int(event.get("seed", -1)) == seed]
    successful = [
        event
        for event in seed_events
        if event.get("event") == "finished" and event.get("exit_status") == 0
    ]
    if shard.is_file():
        if any(event.get("output_sha256") == digest(shard) for event in successful):
            return "completed"
        return "preserved-unverified"
    interruption = out / "execution-interruption.json"
    if interruption.is_file():
        record_data = json.loads(interruption.read_text(encoding="utf-8"))
        if int(record_data.get("seed", -1)) == seed:
            return "interrupted"
    if any(
        event.get("event") in {"started", "interrupted", "finished"}
        for event in seed_events
    ):
        return "interrupted"
    if seed_events or log.exists():
        return "preserved-unverified"
    return "unattempted"


def run_seed(seed: int, *, inventory: Path | None = None) -> None:
    if seed not in SEEDS:
        raise ValueError(f"seed_outside_frozen_plan:{seed}")
    inventory = inventory or CONTINUATION_INVENTORY
    preflight()
    if seed_state(seed) != "unattempted":
        raise RuntimeError(f"generation_refuses_non_unattempted_seed:{seed}")
    target = OUT / f"shard-{seed}.jsonl"
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
    log = OUT / f"shard-{seed}.log"
    if target.exists() or log.exists():
        raise FileExistsError(f"refusing to replace shard output/log for seed {seed}")
    record({**common, "event": "started"}, inventory=inventory)
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
            },
            inventory=inventory,
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
        },
        inventory=inventory,
    )
    if exit_status:
        raise subprocess.CalledProcessError(exit_status, command)


def main(*, execute: bool = False) -> list[dict[str, Any]]:
    frozen = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [{"seed": seed, "state": seed_state(seed)} for seed in SEEDS]
    if not execute:
        return plan
    for item in plan:
        seed, state = item["seed"], item["state"]
        if state == "unattempted":
            run_seed(seed, inventory=CONTINUATION_INVENTORY)
            continue
        record(
            {
                "event": "continuation_skip_preserved",
                "seed": seed,
                "state": state,
                "policy": frozen["resume_policy"]["interrupted_or_ambiguous_attempts"],
                "timestamp": timestamp(),
            },
            inventory=CONTINUATION_INVENTORY,
        )
    return plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch only seeds prospectively classified as unattempted",
    )
    arguments = parser.parse_args()
    print(json.dumps(main(execute=arguments.execute), indent=2, sort_keys=True))
