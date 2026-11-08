"""Launch one seed555 pair and bind its raw report to the frozen contract."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed551_controlled_policy_strength as runtime
from ml.alphazero_lite import seed555_policy_strength_screen as protocol
from ml.alphazero_lite.run_seed555_policy_strength_screen import _read_live_cpu_max


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arena.main()
    argv = sys.argv[1:]
    seed = int(argv[argv.index("--seed") + 1])
    index = seed - protocol.SEED
    report_path = Path(argv[argv.index("--out") + 1])
    output = report_path.parents[1]
    commands = [
        json.loads(line)
        for line in (output / "command-preflight.jsonl").read_text().splitlines()
        if line
    ]
    command = commands[index]
    authentication_path = output / "runtime-authentication.json"
    authentication = json.loads(authentication_path.read_text())
    assets = authentication["runtime_assets"]
    cpu_max_path, cpu_max = _read_live_cpu_max()
    identity = {
        "opening_index": index,
        "seed": seed,
        "command_preflight_entry_sha256": hashlib.sha256(
            protocol.stable_json(command)
        ).hexdigest(),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(authentication_path),
        "component_artifact_sha256": {
            name: value["checkpoint_sha256"]
            for name, value in authentication["components"].items()
        },
        "search_policy_sidecar_sha256": {
            name: value["files"]["search_policy.json"]
            for name, value in authentication["components"].items()
        },
        "tablebase_object_sha256": assets["tablebase_materialized_sha256"],
        "tablebase_object_size_bytes": assets["tablebase_materialized_size_bytes"],
        "tablebase_resolved_runtime_path": assets["tablebase_runtime_path"],
        "tablebase_repository_pointer_sha256": assets["tablebase_repository_sha256"],
        "tablebase_repository_pointer_size_bytes": assets[
            "tablebase_repository_size_bytes"
        ],
        "historical_lfs_pointer_sha256": assets["historical_lfs_pointer_sha256"],
        "historical_lfs_pointer_git_blob": assets["historical_lfs_pointer_git_blob"],
        "native_probe_sha256": assets["native_probe_sha256"],
        "registered_runtime": command["runtime_identity"],
        "observed_runtime": runtime.observed_runtime(),
        "live_cpu_max_path": cpu_max_path,
        "live_cpu_max": cpu_max,
    }
    report = json.loads(report_path.read_text())
    notes = report.setdefault("notes", {})
    profile = notes.setdefault("search_profile", {})
    notes["seed555_execution_identity"] = identity
    profile["seed555_execution_identity"] = identity
    unsigned = {key: value for key, value in profile.items() if key != "hash"}
    profile_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    profile["hash"] = profile_hash
    notes["search_profile_hash"] = profile_hash
    temporary = report_path.with_name(f".{report_path.name}.identity-pending")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, report_path)
    directory_fd = os.open(
        report_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    main()
