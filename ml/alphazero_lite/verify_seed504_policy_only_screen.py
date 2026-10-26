"""Read-only replay and analysis verifier for the policy-only strength screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed504_policy_only_screen as protocol


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
            or command.get("argv")
            != protocol.command(root, index, prepare_opening=False)
            or command.get("registration_sha256")
            != protocol.sha(out / "registration.json")
            or command.get("suite_sha256") != protocol.sha(out / "openings.jsonl")
            or command.get("component_assignments") != registration["assignments"]
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
