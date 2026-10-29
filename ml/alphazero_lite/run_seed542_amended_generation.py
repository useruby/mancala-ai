"""Run only the authenticated, unattempted seed542 amendment sources."""

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

from ml.alphazero_lite import run_seed542_generation as historical
from ml.alphazero_lite import seed542_amendment

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
AMENDMENT = CORPUS / "continuation-amendment.json"
AMENDMENT_SHA256 = "098a7297d7c73cc3ae8701ea285f0dff26917d8d695310edae986bda17be42e4"
INTERPRETER = ROOT / ".tmp/seed542-recorded-runtime/bin/python"
INVENTORY = CORPUS / "continuation-inventory.jsonl"
TABLEBASE_SOURCE = Path("/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb")


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def bound_source_digest(relative: str) -> str | None:
    path = ROOT / relative
    if relative != "ml/alphazero_lite/run_seed542_amended_generation.py":
        return digest(path)
    source = path.read_text(encoding="utf-8")
    normalized, replacements = re.subn(
        r'^AMENDMENT_SHA256 = "[^"]+"$',
        'AMENDMENT_SHA256 = "<frozen-amendment-sha256>"',
        source,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError("amended_runner_self_binding_normalization_failed")
    return hashlib.sha256(normalized.encode()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def actual_environment() -> dict[str, str]:
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def command_for(seed: int, games: int) -> list[str]:
    command = list(historical.COMMAND_TEMPLATE)
    command[0] = str(INTERPRETER)
    game_index = command.index("--games") + 1
    seed_index = command.index("--seed") + 1
    output_index = command.index("--out") + 1
    command[game_index] = str(games)
    command[seed_index] = str(seed)
    command[output_index] = (
        (CORPUS / f"shard-{seed}.jsonl").relative_to(ROOT).as_posix()
    )
    return command


def command_template() -> list[str]:
    template = command_for(542002, 64)
    template[template.index("--games") + 1] = "{games}"
    template[template.index("--seed") + 1] = "{seed}"
    template[template.index("--out") + 1] = "{output}"
    return template


def authenticate_command_template(
    amendment: dict[str, Any], active_template: list[str] | None = None
) -> None:
    frozen = amendment.get("command_template")
    if frozen != command_template() or (
        active_template is not None and active_template != frozen
    ):
        raise RuntimeError("amended_preflight_command_template_drift")


def authenticate_assets(root: Path, assets: dict[str, dict[str, Any]]) -> None:
    for asset in assets.values():
        if digest(root / asset["path"]) != asset["sha256"]:
            raise RuntimeError(
                f"amended_preflight_asset_sha256_mismatch:{asset['path']}"
            )


def preflight(
    *,
    amendment_path: Path = AMENDMENT,
    amendment_sha256: str = AMENDMENT_SHA256,
    active_environment: dict[str, str] | None = None,
    active_template: list[str] | None = None,
) -> dict[str, Any]:
    """Authenticate historical bindings and prospective amendment before writes."""
    original_freeze_path = CORPUS / "continuation-freeze.json"
    if digest(original_freeze_path) != historical.FREEZE_SHA256:
        raise RuntimeError("amended_preflight_historical_freeze_mismatch")
    amendment_bytes = amendment_path.read_bytes()
    if hashlib.sha256(amendment_bytes).hexdigest() != amendment_sha256:
        raise RuntimeError("amended_preflight_amendment_sha256_mismatch")
    amendment = json.loads(amendment_bytes)
    if amendment.get("status") != "frozen continuation amendment":
        raise RuntimeError("amended_preflight_amendment_not_frozen")
    environment = active_environment or actual_environment()
    if environment != amendment["environment"]:
        raise RuntimeError("amended_preflight_environment_drift")
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise RuntimeError("amended_preflight_interpreter_mismatch")
    if not INTERPRETER.is_file() or not os.access(INTERPRETER, os.X_OK):
        raise RuntimeError("amended_preflight_interpreter_unavailable")
    authenticate_command_template(amendment, active_template)
    if amendment["resource_scope"] != historical.RESOURCE_SCOPE:
        raise RuntimeError("amended_preflight_resource_scope_drift")
    if amendment["amendment"]["supplement"] != {
        "seed": 542064,
        "games": 13,
        "source": "training",
        "same_behavior_and_targets_as_original": True,
    }:
        raise RuntimeError("amended_preflight_supplement_registration_drift")
    for relative, expected in amendment["source_sha256"].items():
        if bound_source_digest(relative) != expected:
            raise RuntimeError(f"amended_preflight_source_sha256_mismatch:{relative}")
    authenticate_assets(ROOT, amendment["assets"])
    tablebase = amendment["assets"]["tablebase"]
    if (
        Path(tablebase["existing_local_source"]).resolve() != TABLEBASE_SOURCE.resolve()
        or digest(TABLEBASE_SOURCE) != tablebase["sha256"]
        or TABLEBASE_SOURCE.stat().st_size != tablebase["size_bytes"]
    ):
        raise RuntimeError("amended_preflight_existing_tablebase_source_mismatch")
    preserved = amendment["preserved_evidence"]
    preserved_files = {
        "docs/data/seed542-selfplay-4096/manifest.md": "original_manifest_sha256",
        "docs/data/seed542-selfplay-4096/continuation-freeze.json": "continuation_freeze_sha256",
        "docs/data/seed542-selfplay-4096/execution-interruption.json": "interruption_record_sha256",
        "docs/data/seed542-selfplay-4096/execution-inventory.jsonl": "execution_inventory_sha256",
        "docs/data/seed542-selfplay-4096/accounting.json": "original_accounting_sha256",
        "docs/data/seed542-selfplay-4096/partial-status.json": "original_partial_status_sha256",
        "docs/data/seed542-selfplay-4096/shard-542000.jsonl": "seed542000_sha256",
        "docs/data/seed542-selfplay-4096/shard-542000.log": "seed542000_log_sha256",
        "docs/data/seed542-selfplay-4096/shard-542001.log": "seed542001_log_sha256",
        "docs/data/seed542-selfplay-4096/azlite-self-play-shards-etoy6j_i/worker_0.jsonl": "staging_sha256",
    }
    for relative, key in preserved_files.items():
        if digest(ROOT / relative) != preserved[key]:
            raise RuntimeError(
                f"amended_preflight_preserved_evidence_changed:{relative}"
            )
    interruption = json.loads((CORPUS / "execution-interruption.json").read_text())
    staging = ROOT / interruption["staging_output"]["path"]
    if digest(staging) != interruption["staging_output"]["sha256"]:
        raise RuntimeError("amended_preflight_interrupted_evidence_changed")
    if digest(staging) != preserved["staging_sha256"]:
        raise RuntimeError("amended_preflight_staging_binding_changed")
    recovered = amendment["recovered_derivative"]
    extraction = seed542_amendment.verify_extraction(
        staging,
        ROOT / recovered["path"],
        ROOT / recovered["provenance_path"],
        source_path_label=recovered["original_source_path"],
    )
    if (
        extraction["derivative_sha256"] != recovered["sha256"]
        or extraction["provenance_sha256"] != recovered["provenance_sha256"]
        or extraction["complete_games"] != 51
        or extraction["rows"] != 1768
    ):
        raise RuntimeError("amended_preflight_recovered_derivative_mismatch")
    for seed in range(542000, 542064):
        if seed == 542001:
            if historical.seed_state(seed, root=ROOT) != "interrupted":
                raise RuntimeError("amended_preflight_interruption_state_changed")
        elif seed == 542000:
            if historical.seed_state(seed, root=ROOT) != "completed":
                raise RuntimeError("amended_preflight_completed_shard_state_changed")
    return amendment


def record(event: dict[str, Any]) -> None:
    with INVENTORY.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def amended_seed_state(seed: int) -> str:
    if seed == 542000 or seed == 542001:
        return historical.seed_state(seed, root=ROOT)
    output = CORPUS / f"shard-{seed}.jsonl"
    log = CORPUS / f"shard-{seed}.log"
    events = historical.inventory_events(CORPUS / "continuation-inventory.jsonl")
    seed_events = [event for event in events if int(event.get("seed", -1)) == seed]
    for event in reversed(seed_events):
        if event.get("event") == "finished":
            expected_games = 13 if seed == 542064 else 64
            if (
                event.get("exit_status") == 0
                and event.get("completed_games") == expected_games
                and event.get("capped_games") == 0
                and event.get("incomplete_or_unstarted_games") == 0
                and event.get("output_sha256") == digest(output)
            ):
                return "completed"
            return "preserved-unverified"
        if event.get("event") in {"started", "interrupted"}:
            return "interrupted"
    if output.exists() or log.exists():
        return "preserved-unverified"
    return "unattempted"


def _completion(path: Path, requested: int) -> tuple[int, int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["game_index"]), []).append(row)
    completed = capped = incomplete = 0
    for game_rows in grouped.values():
        ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
        expected = len(ordered[0].get("game_moves", []))
        if any(row.get("game_capped") is True for row in ordered):
            capped += 1
        elif (
            len(ordered) == expected
            and [int(row["move_index"]) for row in ordered] == list(range(expected))
            and all(row.get("game_completed") is True for row in ordered)
        ):
            completed += 1
        else:
            incomplete += 1
    return completed, capped, requested - completed - capped - incomplete


def run_one(seed: int, games: int, *, role: str) -> None:
    preflight()
    command = command_for(seed, games)
    output = CORPUS / f"shard-{seed}.jsonl"
    log = CORPUS / f"shard-{seed}.log"
    if output.exists() or log.exists():
        raise FileExistsError(f"amended_generation_refuses_existing_output:{seed}")
    started = time.monotonic()
    common = {
        "seed": seed,
        "source": role,
        "requested_games": games,
        "command": command,
        "command_text": " ".join(command),
        "started_at": timestamp(),
        "environment": actual_environment(),
        "tablebase_source": str(TABLEBASE_SOURCE),
    }
    record({**common, "event": "started"})
    try:
        with log.open("xb") as stream:
            process = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
    except BaseException as error:
        record(
            {
                **common,
                "event": "interrupted",
                "ended_at": timestamp(),
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(error),
                "output_bytes": output.stat().st_size if output.exists() else 0,
                "output_sha256": digest(output),
                "log_sha256": digest(log),
            }
        )
        raise
    completion = _completion(output, games) if output.is_file() else (0, 0, 0)
    record(
        {
            **common,
            "event": "finished",
            "ended_at": timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "exit_status": process.returncode,
            "completed_games": completion[0],
            "capped_games": completion[1],
            "incomplete_or_unstarted_games": completion[2],
            "output_bytes": output.stat().st_size if output.exists() else 0,
            "output_sha256": digest(output),
            "log_sha256": digest(log),
        }
    )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    if completion != (games, 0, 0):
        raise RuntimeError(
            f"amended_generation_noncomplete_outcome:{seed}:{completion}"
        )


def main(*, execute: bool = False) -> list[dict[str, Any]]:
    amendment = preflight()
    plan = [
        {"seed": seed, "games": 64, "state": amended_seed_state(seed)}
        for seed in range(542000, 542064)
    ]
    plan.append(
        {
            "seed": 542064,
            "games": 13,
            "state": amended_seed_state(542064),
        }
    )
    if not execute:
        return plan
    for item in plan:
        seed = item["seed"]
        if seed == 542001 or item["state"] == "completed":
            record(
                {
                    "event": "amendment_skip_preserved",
                    "seed": seed,
                    "state": item["state"],
                    "timestamp": timestamp(),
                }
            )
            continue
        if item["state"] != "unattempted":
            raise RuntimeError(
                f"amended_generation_refuses_non_unattempted_seed:{seed}"
            )
        role = "development" if 542048 <= seed <= 542063 else "training"
        run_one(seed, item["games"], role=role)
    if amendment.get("amendment", {}).get("supplement", {}).get("seed") != 542064:
        raise RuntimeError("amended_supplement_registration_missing")
    return plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(execute=args.execute), indent=2, sort_keys=True))
