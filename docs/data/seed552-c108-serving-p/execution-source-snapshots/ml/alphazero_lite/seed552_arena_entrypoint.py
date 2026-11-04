"""Initialize PyTorch threads before one seed552 arena invocation."""

import hashlib
import json
import sys
from pathlib import Path

import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed551_controlled_policy_strength as runtime
from ml.alphazero_lite import seed552_c108_serving_p as protocol


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arena.main()
    argv = sys.argv[1:]
    seed = int(argv[argv.index("--seed") + 1])
    index = seed - protocol.SEED
    root = Path.cwd()
    output = root / protocol.OUT
    command = [
        json.loads(line)
        for line in (output / "command-preflight.jsonl").read_text().splitlines()
        if line
    ][index]
    auth_path = output / "runtime-authentication.json"
    auth = json.loads(auth_path.read_text())
    identity = {
        "opening_index": index,
        "seed": seed,
        "command_preflight_entry_sha256": hashlib.sha256(
            json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": hashlib.sha256(auth_path.read_bytes()).hexdigest(),
        "tablebase_object_sha256": auth["tablebase"]["materialized_object_sha256"],
        "tablebase_object_size_bytes": auth["tablebase"]["materialized_object_size_bytes"],
        "tablebase_resolved_runtime_path": auth["tablebase"]["resolved_runtime_path"],
        "native_probe_sha256": auth["native_probe"]["sha256"],
        "registered_runtime": command["runtime_identity"],
        "observed_runtime": runtime.observed_runtime(),
    }
    report_path = Path(argv[argv.index("--out") + 1])
    report = json.loads(report_path.read_text())
    notes = report.setdefault("notes", {})
    profile = notes.setdefault("search_profile", {})
    notes["seed552_execution_identity"] = identity
    profile["seed552_execution_identity"] = identity
    unhashed_profile = {key: value for key, value in profile.items() if key != "hash"}
    profile_hash = hashlib.sha256(
        json.dumps(unhashed_profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    profile["hash"] = profile_hash
    notes["search_profile_hash"] = profile_hash
    pending = report_path.with_name(f".{report_path.name}.identity-pending")
    pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pending.replace(report_path)


if __name__ == "__main__":
    main()
