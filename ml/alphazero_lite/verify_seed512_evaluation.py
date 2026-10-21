"""Independent replay and result verification for seed512."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed512_evaluation as protocol


def independent_analysis(root: Path, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the primary gate and matched-opening contrast from raw ledger rows."""
    grouped: dict[int, list[float]] = {i: [] for i in range(1024)}
    for row in ledger:
        result = row["winner"]
        grouped[int(row["opening_index"])].append(
            1.0 if result == "challenger" else 0.5 if result == "draw" else 0.0
        )
    if len(ledger) != 2048 or any(len(values) != 2 for values in grouped.values()):
        raise ValueError("seed512_independent_pair_count")
    current = np.fromiter((sum(grouped[i]) / 2 for i in range(1024)), dtype=float)
    old_groups: dict[int, list[float]] = {i: [] for i in range(1024)}
    old_rows = protocol.read_rows(root / protocol.SEED511_OUT / "outcomes.jsonl")
    for row in old_rows:
        val = (
            1.0
            if row["winner"] == "challenger"
            else 0.5
            if row["winner"] == "draw"
            else 0.0
        )
        old_groups[int(row["opening_index"])].append(val)
    if any(len(old_groups[i]) != 2 for i in range(1024)):
        raise ValueError("seed512_seed511_pair_count")
    old = np.fromiter((sum(old_groups[i]) / 2 for i in range(1024)), dtype=float)
    rng = np.random.default_rng(512)
    picks = rng.integers(0, 1024, size=(20_000, 1024))
    bootstrap = current[picks].mean(axis=1)
    contrast = (current - old)[picks].mean(axis=1)
    wins = sum(row["winner"] == "challenger" for row in ledger)
    draws = sum(row["winner"] == "draw" for row in ledger)
    lower = float(np.quantile(bootstrap, 0.05))
    return {
        "games": len(ledger),
        "wins": wins,
        "draws": draws,
        "losses": 2048 - wins - draws,
        "score": (wins + 0.5 * draws) / 2048,
        "bootstrap_resamples": 20_000,
        "bootstrap_seed": 512,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "pass": lower > 0.5,
        "promotion": False,
        "fresh_confirmation": False,
        "interpretation": "examined exploratory follow-up only; seed511 and its 128-simulation outcomes were already examined",
        "seed511_matched_contrast": {
            "mean_1200_minus_128": float((current - old).mean()),
            "descriptive_two_sided_95": [
                float(np.quantile(contrast, 0.025)),
                float(np.quantile(contrast, 0.975)),
            ],
            "bootstrap_resamples": 20_000,
            "bootstrap_seed": 512,
            "resampling_unit": "paired opening, keeping both seats together",
            "primary_gate_substitute": False,
        },
    }


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    preflight = protocol.preflight(root, allow_started=True)
    out = root / protocol.OUT
    openings = protocol.read_rows(out / "openings.jsonl")
    ledger = protocol.read_rows(out / "outcomes.jsonl")
    if len(ledger) != 2048:
        raise ValueError("seed512_outcome_count")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    if set(grouped) != set(range(1024)):
        raise ValueError("seed512_opening_coverage")
    for index in range(1024):
        protocol.verify_pair(root, index, openings[index], grouped[index])
    commands = protocol.read_rows(out / "commands.jsonl")
    if len(commands) != 1024:
        raise ValueError("seed512_command_count")
    seen_commands: set[int] = set()
    expected_assignments = {
        "policy": "seed504 E4, own trunk",
        "value": "initializer, own trunk",
        "opponent": "complete frozen initializer",
    }
    for row in commands:
        index = int(row["opening_index"])
        if index in seen_commands or row["argv"] != protocol.command(root, index):
            raise ValueError(f"seed512_command_mismatch:{index}")
        seen_commands.add(index)
        if (
            row["registration_sha256"] != preflight["registration_sha256"]
            or row["suite_sha256"] != preflight["suite_sha256"]
            or row["component_assignments"] != expected_assignments
        ):
            raise ValueError(f"seed512_command_binding:{index}")
    if seen_commands != set(range(1024)):
        raise ValueError("seed512_command_coverage")
    computed = independent_analysis(root, ledger)
    published = json.loads((out / "results.json").read_text())
    if computed != published:
        raise ValueError("seed512_result_mismatch")
    return {
        "status": "verified_complete",
        "preflight": preflight,
        "games_replayed": 2048,
        "results": computed,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
