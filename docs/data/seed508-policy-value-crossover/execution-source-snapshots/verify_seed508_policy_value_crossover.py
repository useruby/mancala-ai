"""Independent game replay and runtime/source verifier for seed508."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed508_policy_value_crossover as protocol
from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game


def verify_complete_pair(
    root: Path,
    arm: str,
    index: int,
    opening: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> None:
    """Verify a committed pair's receipt binding, assignments, runtime, and replay."""
    protocol.validate_pair(root, arm, index, opening, ledger)


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    reg = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if protocol.sha(out / "registration.json") != freeze.get("registration_sha256"):
        raise ValueError("registration_freeze_mismatch")
    if protocol.sha(out / "openings.jsonl") != reg["suite"]["sha256"]:
        raise ValueError("suite_binding")
    openings = protocol.rows(out / "openings.jsonl")
    if len(openings) != 512:
        raise ValueError("suite_count")
    for name, digest in reg["sources_sha256"].items():
        if protocol.sha(root / name) != digest:
            raise ValueError(f"source_binding:{name}")
    for path, digest in reg["runtime"]["runtime_identities_sha256"].items():
        if protocol.sha(root / path) != digest:
            raise ValueError(f"runtime_identity:{path}")
    if protocol.sha(root / "docs/data/seed504/seed504_epoch4.npz") != protocol.E4_SHA:
        raise ValueError("e4_checkpoint_identity")
    if protocol.sha(out / "initializer.npz") != protocol.INIT_SHA:
        raise ValueError("initializer_checkpoint_identity")
    for name, component in reg["components"].items():
        for filename, digest in component["files"].items():
            if protocol.sha(out / component["artifact"] / filename) != digest:
                raise ValueError(f"component_artifact:{name}:{filename}")
    ledger = protocol.rows(out / "outcomes.jsonl")
    if len(ledger) != 2048:
        raise ValueError("game_count")
    grouped = {(arm, i): [] for arm in protocol.ARMS for i in range(512)}
    for row in ledger:
        key = (str(row["arm"]), int(row["opening_index"]))
        if key not in grouped:
            raise ValueError("outcome_assignment")
        grouped[key].append(row)
    for arm in protocol.ARMS:
        for i, opening in enumerate(openings):
            pair = grouped[(arm, i)]
            if len(pair) != 2 or {int(r["challenger_player"]) for r in pair} != {0, 1}:
                raise ValueError(f"paired_seats:{arm}:{i}")
            batch = out / "batches"
            op = batch / f"{arm}-opening-{i:04d}.jsonl"
            gp = batch / f"{arm}-games-{i:04d}.jsonl"
            rp = batch / f"{arm}-report-{i:04d}.json"
            games, report = protocol.rows(gp), json.loads(rp.read_text())
            notes = report.get("notes", {})
            if (
                op.read_bytes() != base.canonical_json(opening) + b"\n"
                or len(games) != 2
                or report.get("games_played") != 2
            ):
                raise ValueError(f"batch_shape:{arm}:{i}")
            if (
                notes.get("challenger_simulations") != 128
                or notes.get("current_simulations") != 128
                or notes.get("workers_used") != 1
                or notes.get("base_seed") != 488 + i
                or notes.get("seed_contract") != "azlite_eval_seed_v2"
                or notes.get("search_profile", {}).get("c_puct") != 1.25
                or notes.get("exact_root_solve_threshold") != 16
            ):
                raise ValueError(f"batch_runtime:{arm}:{i}")
            command_record = next(
                (
                    r
                    for r in protocol.rows(out / "commands.jsonl")
                    if r["arm"] == arm and int(r["opening_index"]) == i
                ),
                None,
            )
            if command_record is None:
                raise ValueError(f"command_missing:{arm}:{i}")
            cmd = command_record["argv"]
            expected = protocol.ARMS[arm]
            for flag, component in (
                ("--challenger-policy-artifact", expected["policy"]),
                ("--challenger-value-artifact", expected["value"]),
            ):
                if cmd[cmd.index(flag) + 1] != str(
                    out / "runtime-artifacts" / component
                ):
                    raise ValueError(f"command_assignment:{arm}:{i}:{flag}")
            verify_complete_pair(root, arm, i, opening, ledger)
            for row in pair:
                game = games[int(row["batch_game_position"])]
                _replay_game(opening, game, int(row["challenger_player"]))
    results = protocol.analyze(ledger)
    published = json.loads((out / "results.json").read_text())
    if published != results:
        raise ValueError("published_result_mismatch")
    return {
        "status": "verified_complete",
        "games": len(ledger),
        "primary": results["primary"],
        "standalone": results["standalone"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
