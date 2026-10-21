"""Verify a physical checkout copy of the seed512 publication bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import seed512_evaluation as protocol  # noqa: E402
from ml.alphazero_lite.verify_seed512_evaluation import (  # noqa: E402
    independent_analysis,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _portable(argv: list[str]) -> list[str]:
    return [Path(token).name if token.startswith("/") else token for token in argv]


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    receipt_path = out / "publication-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    actual = {
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path != receipt_path
    }
    if actual != set(receipt["files"]) or len(actual) != receipt["file_count"]:
        raise ValueError("seed512_copy_file_set")
    for rel, digest in receipt["files"].items():
        if sha(out / rel) != digest:
            raise ValueError(f"seed512_copy_hash:{rel}")
    protocol.preflight(root, allow_started=True)
    openings, ledger = rows(out / "openings.jsonl"), rows(out / "outcomes.jsonl")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    if len(openings) != 1024 or len(ledger) != 2048 or set(grouped) != set(range(1024)):
        raise ValueError("seed512_copy_coverage")
    for index, opening in enumerate(openings):
        protocol.verify_pair(root, index, opening, grouped[index])
    commands = rows(out / "commands.jsonl")
    if len(commands) != 1024 or {int(row["opening_index"]) for row in commands} != set(
        range(1024)
    ):
        raise ValueError("seed512_copy_command_coverage")
    for row in commands:
        index = int(row["opening_index"])
        if _portable(row["argv"]) != _portable(protocol.command(root, index)):
            raise ValueError(f"seed512_copy_command:{index}")
    result = independent_analysis(root, ledger)
    if result != json.loads((out / "results.json").read_text()):
        raise ValueError("seed512_copy_analysis")
    return {
        "status": "physical_copy_verified",
        "games_replayed": len(ledger),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
