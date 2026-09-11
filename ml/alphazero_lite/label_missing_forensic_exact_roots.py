#!/usr/bin/env python3
"""Label only unresolved forensic roots with the pinned tier-21 native oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.forensic_exact_references import sha256_file  # noqa: E402
from ml.alphazero_lite.forensic_suite import load_suite  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    NativeHybridProcess,
    native_label_payload,
    root_training_value,
    stable_sha,
    validate_native_response,
)

TIER21_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"


def solve(
    position: Any, probe: Path, tablebase: Path, timeout: float
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "canonical_state": position.canonical_key,
        "id": position.id,
        "status": "exact_error",
        "timeout_seconds": timeout,
        "probe_sha256": sha256_file(probe),
        "tablebase_sha256": sha256_file(tablebase),
        "tablebase_tier": 21,
    }
    process = NativeHybridProcess(probe, tablebase)
    try:
        native = process.request(native_label_payload(position.state), timeout)
        game = KalahGame.from_state(position.state)
        values = {
            int(move): int(value) for move, value in native["action_values"].items()
        }
        exact_value = validate_native_response(
            values,
            native["optimal_actions"],
            int(native["exact_value"]),
            game.possible_moves(),
            game.current_player,
        )
        result.update(
            {
                "status": "exact_solved",
                "exact_value": exact_value,
                "exact_action_values": values,
                "exact_optimal_actions": sorted(
                    int(move) for move in native["optimal_actions"]
                ),
                "exact_root_value": root_training_value(
                    exact_value, game.current_player
                ),
                "native_metrics": native.get("metrics", {}),
            }
        )
    except TimeoutError as error:
        result.update({"status": "exact_timeout", "failure_reason": str(error)})
    except (
        Exception
    ) as error:  # Native failures must be explicit, never MCTS fallbacks.
        result.update({"status": "exact_error", "failure_reason": str(error)})
    finally:
        process.close()
    result["wall_duration_seconds"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if sha256_file(args.tablebase) != TIER21_SHA256:
        raise SystemExit("tier-21 tablebase SHA mismatch")
    existing = json.loads(args.reference.read_text())["rows"]
    unresolved = {
        row["canonical_state"]
        for row in existing
        if row["exact_status"] == "unresolved"
    }
    positions = [
        row for row in load_suite(args.suite) if row.canonical_key in unresolved
    ]
    positions.sort(key=lambda row: stable_sha(row.canonical_key))
    completed = {}
    if args.out.exists():
        completed = {
            row["canonical_state"]: row
            for row in json.loads(args.out.read_text()).get("rows", [])
        }
    pending = [row for row in positions if row.canonical_key not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                solve, row, args.native_probe, args.tablebase, args.timeout_seconds
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = future.result()
            completed[row["canonical_state"]] = row
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_missing_root_labels_v1",
                        "rows": [
                            completed[row.canonical_key]
                            for row in positions
                            if row.canonical_key in completed
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
