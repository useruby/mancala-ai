"""Independent replay and publication verifier for seed505."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed505_strength_screen as protocol
from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    reg = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if freeze.get("status") != "frozen_before_execution" or protocol.sha(out / "registration.json") != freeze.get("registration_sha256"):
        raise ValueError("freeze_binding")
    if protocol.sha(out / "openings.jsonl") != reg["suite_reuse"]["suite_sha256"] or len(protocol.read_rows(out / "openings.jsonl")) != 512:
        raise ValueError("suite_binding")
    for path, digest in reg["source_sha256"].items():
        if protocol.sha(root / path) != digest:
            raise ValueError(f"source_binding:{path}")
    for path, digest in reg["runtime"]["runtime_identities_sha256"].items():
        if protocol.sha(root / path) != digest:
            raise ValueError(f"runtime_binding:{path}")
    if protocol.sha(out / "initializer.npz") != protocol.INITIALIZER_SHA256:
        raise ValueError("initializer_binding")
    if protocol.sha(root / protocol.SEED504 / "seed504_epoch4.npz") != protocol.E4_SHA256:
        raise ValueError("e4_binding")
    artifacts = json.loads((out / "runtime-artifacts.json").read_text())
    if protocol.sha(out / "runtime-artifacts.json") != freeze.get("artifact_manifest_sha256"):
        raise ValueError("artifact_manifest_binding")
    import numpy as np
    for arm, checkpoint in (("E4", root / protocol.SEED504 / "seed504_epoch4.npz"), ("initializer", out / "initializer.npz")):
        folder = out / "runtime-artifacts" / arm
        for name, digest in artifacts[arm].items():
            if protocol.sha(folder / name) != digest:
                raise ValueError(f"artifact_hash:{arm}:{name}")
        with np.load(checkpoint, allow_pickle=False) as expected, np.load(folder / "model.npz", allow_pickle=False) as actual:
            if set(expected.files) != set(actual.files) or any(not np.array_equal(expected[k], actual[k]) for k in expected.files):
                raise ValueError(f"export_parameters:{arm}")
        if (folder / "search_policy.json").read_bytes() != (root / "model-artifact/current/search_policy.json").read_bytes():
            raise ValueError(f"search_policy:{arm}")
    openings = protocol.read_rows(out / "openings.jsonl")
    rows = protocol.read_rows(out / "outcomes.jsonl")
    if len(rows) != protocol.GAME_COUNT:
        raise ValueError("game_count")
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["opening_index"]), []).append(row)
    if set(groups) != set(range(512)):
        raise ValueError("opening_coverage")
    for i, opening in enumerate(openings):
        pair = groups[i]
        if len(pair) != 2 or {int(r["F_player"]) for r in pair} != {0, 1}:
            raise ValueError(f"paired_seats:{i}")
        folder = out / "batches"
        opener = folder / f"opening-{i:04d}.jsonl"
        games_path = folder / f"games-{i:04d}.jsonl"
        report_path = folder / f"report-{i:04d}.json"
        games = protocol.read_rows(games_path)
        report = json.loads(report_path.read_text()); notes = report.get("notes", {})
        if opener.read_bytes() != base.canonical_json(opening) + b"\n" or len(games) != 2 or report.get("games_played") != 2:
            raise ValueError(f"batch_shape:{i}")
        if notes.get("challenger_simulations") != 128 or notes.get("current_simulations") != 128 or notes.get("workers_used") != 1 or notes.get("base_seed") != 488 + i or notes.get("seed_contract") != "azlite_eval_seed_v2" or notes.get("search_profile", {}).get("c_puct") != 1.25 or notes.get("exact_root_solve_threshold") != 16:
            raise ValueError(f"batch_runtime:{i}")
        for row in pair:
            p = int(row["batch_game_position"]); game = games[p]
            if protocol.sha(games_path) != row["batch_game_sha256"] or protocol.sha(report_path) != row["batch_report_sha256"] or protocol.sha(opener) != row["batch_opening_sha256"]:
                raise ValueError(f"batch_hash:{i}")
            replayed, _, _ = _replay_game(opening, game, int(row["F_player"]))
            normalized = {"challenger": "challenger", "current": "current"}.get(replayed, replayed)
            if int(game["challenger_player"]) != int(row["F_player"]) or normalized != row["winner"]:
                raise ValueError(f"replay_outcome:{i}:{p}")
    result = json.loads((out / "results.json").read_text())
    wins = sum(row["winner"] == "challenger" for row in rows); draws = sum(row["winner"] == "draw" for row in rows); losses = sum(row["winner"] == "current" for row in rows)
    if (wins, draws, losses) != (result["wins"], result["draws"], result["losses"]):
        raise ValueError("result_counts")
    return {"status": "verified_complete", "games": len(rows), "wins": wins, "draws": draws, "losses": losses, "score": result["score"], "pass": result["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
