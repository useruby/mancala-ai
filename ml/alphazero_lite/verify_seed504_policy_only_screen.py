"""Read-only replay and analysis verifier for the policy-only strength screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed504_policy_only_screen as protocol


def _relocate_command(argv: list[str], root: Path) -> list[str]:
    """Resolve registered absolute checkout paths in a physical copy."""
    result = list(argv)
    try:
        artifact = Path(result[result.index("--challenger") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("command_challenger_path_missing") from error
    suffix = protocol.OUT / "runtime-artifacts/T-E4"
    if tuple(artifact.parts[-len(suffix.parts) :]) != suffix.parts:
        raise ValueError("command_artifact_path_mismatch")
    original_root = artifact.parents[len(suffix.parts) - 1]
    original = str(original_root)
    replacement = str(root.resolve())
    return [
        result[0],
        *[
            replacement + item[len(original) :]
            if item == original or item.startswith(original + "/")
            else item
            for item in result[1:]
        ],
    ]


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    registration = protocol.preflight(root, allow_started=True)
    out = root / protocol.OUT
    openings = protocol.read_rows(out / "openings.jsonl")
    ledger = (
        protocol.read_rows(out / "outcomes.jsonl")
        if (out / "outcomes.jsonl").exists()
        else []
    )
    committed = protocol._verify_committed(root, openings, ledger)
    commands = (
        protocol.read_rows(out / "commands.jsonl")
        if (out / "commands.jsonl").exists()
        else []
    )
    for command in commands:
        index = int(command["opening_index"])
        if (
            index not in range(protocol.OPENINGS)
            or _relocate_command(command.get("argv", []), root)
            != protocol.command(root, index, prepare_opening=False)
            or command.get("registration_sha256")
            != protocol.sha(out / "registration.json")
            or command.get("suite_sha256") != protocol.sha(out / "openings.jsonl")
            or command.get("seed") != protocol.BASE_SEED + index
            or command.get("component_assignments") != registration["assignments"]
            or command.get("component_files")
            != {
                key: value["files"] for key, value in registration["components"].items()
            }
        ):
            raise ValueError(f"command_binding_mismatch:{index}")
    complete = len(committed) == protocol.OPENINGS and len(ledger) == protocol.GAMES
    if not complete:
        return {
            "status": "verified_incomplete",
            "validated_pairs": len(committed),
            "validated_games": len(ledger),
            "advance": False,
        }
    actual = protocol.analyze(ledger)
    published = json.loads((out / "results.json").read_text())
    if published != actual:
        raise ValueError("result_analysis_mismatch")
    return {
        "status": "verified_complete",
        "validated_pairs": len(committed),
        "validated_games": len(ledger),
        "results": actual,
        "advance": actual["advance"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
