"""Read-only physical-checkout verification for the published seed511 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import seed511_fixed_composition_screen as protocol  # noqa: E402
from ml.alphazero_lite.verify_seed511_fixed_composition_screen import (  # noqa: E402
    analyze_independently,
    verify_pair,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _arg(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"command_argument:{name}") from error


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    receipt_path = out / "publication-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    actual_files = {
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path != receipt_path
    }
    if actual_files != set(receipt["files"]) or len(actual_files) != receipt["file_count"]:
        raise ValueError("copy_receipt_file_set")
    for rel, digest in receipt["files"].items():
        if sha(out / rel) != digest:
            raise ValueError(f"copy_receipt_hash:{rel}")
    if (
        sha(out / "registration.json") != receipt["registration_sha256"]
        or sha(out / "execution-freeze.json") != receipt["execution_freeze_sha256"]
        or sha(out / "openings.jsonl") != receipt["suite_sha256"]
        or sha(out / "outcomes.jsonl") != receipt["outcomes_sha256"]
        or sha(out / "results.json") != receipt["results_sha256"]
        or sha(out / "verification.json") != receipt["verification_sha256"]
    ):
        raise ValueError("copy_receipt_summary_hash")
    preflight = protocol.preflight(root, require_unstarted=False)
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    expected_assignments = {
        "challenger_policy": "E4",
        "challenger_value": "initializer",
        "challenger_policy_trunk": "E4 own trunk",
        "challenger_value_trunk": "initializer own trunk",
        "opponent": "complete initializer artifact",
    }
    command_rows = read_rows(out / "commands.jsonl")
    if not command_rows or {int(row["opening_index"]) for row in command_rows} != set(range(1024)):
        raise ValueError("copy_command_coverage")
    for row in command_rows:
        index = int(row["opening_index"])
        argv = row["argv"]
        options = protocol.base.arena_search_overrides(registration)
        if (
            row["component_assignments"] != expected_assignments
            or row["component_hashes"] != registration["components"]
            or row["runtime"] != registration["runtime"]
            or row["registration_sha256"] != freeze["registration_sha256"]
            or row["suite_sha256"] != freeze["suite_sha256"]
            or Path(_arg(argv, "--challenger")).name != "E4"
            or Path(_arg(argv, "--current")).name != "initializer"
            or Path(_arg(argv, "--challenger-policy-artifact")).name != "E4"
            or Path(_arg(argv, "--challenger-value-artifact")).name != "initializer"
            or Path(_arg(argv, "--exact-root-native-probe")).name != "kalah_v1_tablebase"
            or Path(_arg(argv, "--exact-root-tablebase")).name != "kalah_v1_21.kvtb"
            or json.loads(_arg(argv, "--challenger-search-options-json")) != options
            or json.loads(_arg(argv, "--current-search-options-json")) != options
            or _arg(argv, "--challenger-simulations") != "128"
            or _arg(argv, "--current-simulations") != "128"
            or _arg(argv, "--workers") != "1"
            or _arg(argv, "--c-puct") != "1.25"
            or _arg(argv, "--seed") != str(511 + index)
            or _arg(argv, "--max-moves") != "200"
            or _arg(argv, "--random-opening-plies") != "0"
            or _arg(argv, "--suite-sha256") != sha(out / "batches" / f"opening-{index:04d}.jsonl")
            or Path(_arg(argv, "--opening-prefixes-jsonl")).name != f"opening-{index:04d}.jsonl"
            or Path(_arg(argv, "--game-jsonl")).name != f"games-{index:04d}.jsonl"
            or Path(_arg(argv, "--out")).name != f"report-{index:04d}.json"
        ):
            raise ValueError(f"copy_command_contract:{index}")
    openings = read_rows(out / "openings.jsonl")
    ledger = read_rows(out / "outcomes.jsonl")
    if len(ledger) != 2048:
        raise ValueError("copy_ledger_count")
    for index, opening in enumerate(openings):
        verify_pair(root, index, opening, ledger)
    independently_recomputed = analyze_independently(ledger)
    published = json.loads((out / "results.json").read_text())
    if independently_recomputed != published:
        raise ValueError("copy_result_recompute_mismatch")
    return {
        "status": "physical_copy_verified",
        "preflight": preflight,
        "games_replayed": len(ledger),
        "result": independently_recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
