"""Read-only supplemental semantic verification for archived seed505."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

OUT = Path("docs/data/seed505-strength-screen-v3")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _expect(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def verify(root: Path, out: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    out = out or root / OUT
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    _expect(
        sha(out / "registration.json") == freeze["registration_sha256"],
        "registration_binding",
    )
    for relative, digest in registration["source_sha256"].items():
        source = root / relative
        if not source.is_file() or sha(source) != digest:
            source = out / "execution-source-snapshots" / Path(relative).name
        _expect(
            source.is_file() and sha(source) == digest, f"source_identity:{relative}"
        )
    runtime_identities = registration["runtime"]["runtime_identities_sha256"]
    for relative, digest in runtime_identities.items():
        _expect(
            (root / relative).is_file() and sha(root / relative) == digest,
            f"runtime_identity:{relative}",
        )
    artifact_manifest = json.loads((out / "runtime-artifacts.json").read_text())
    _expect(
        artifact_manifest == registration["runtime_artifacts"],
        "artifact_manifest_registration",
    )
    for arm, manifest in artifact_manifest.items():
        for filename, digest in manifest.items():
            _expect(
                sha(out / "runtime-artifacts" / arm / filename) == digest,
                f"artifact_identity:{arm}:{filename}",
            )
    _expect(
        sha(out / "openings.jsonl") == registration["suite_reuse"]["suite_sha256"],
        "suite_binding",
    )
    openings = rows(out / "openings.jsonl")
    ledger = rows(out / "outcomes.jsonl")
    commands = rows(out / "commands.jsonl")
    _expect(
        len(openings) == 512 and len(ledger) == 1024 and len(commands) == 512,
        "coverage_shape",
    )
    expected_options = registration["runtime"]["search_options"]
    command_by_index = {int(row["opening_index"]): row for row in commands}
    _expect(set(command_by_index) == set(range(512)), "command_coverage")
    points = np.zeros((512, 2), dtype=float)
    for index, opening in enumerate(openings):
        _expect(int(opening["opening_index"]) == index, f"opening_index:{index}")
        batch = out / "batches"
        opener = batch / f"opening-{index:04d}.jsonl"
        game_path = batch / f"games-{index:04d}.jsonl"
        report_path = batch / f"report-{index:04d}.json"
        _expect(
            opener.read_bytes() == base.canonical_json(opening) + b"\n",
            f"opening_artifact:{index}",
        )
        game_rows = rows(game_path)
        report = json.loads(report_path.read_text())
        notes = report["notes"]
        command = command_by_index[index]
        argv = command["argv"]
        options = dict(zip(argv, argv[1:]))
        expected = {
            "--challenger-simulations": "128",
            "--current-simulations": "128",
            "--workers": "1",
            "--c-puct": "1.25",
            "--seed-contract": "azlite_eval_seed_v2",
            "--random-opening-plies": "0",
            "--max-moves": "200",
            "--seed": str(488 + index),
            "--games": "2",
            "--challenger": str(out / "runtime-artifacts/E4"),
            "--current": str(out / "runtime-artifacts/initializer"),
            "--opening-prefixes-jsonl": str(opener),
            "--game-jsonl": str(game_path),
            "--out": str(report_path),
        }
        path_suffixes = {
            "--challenger": Path("runtime-artifacts/E4"),
            "--current": Path("runtime-artifacts/initializer"),
            "--opening-prefixes-jsonl": Path("batches") / opener.name,
            "--game-jsonl": Path("batches") / game_path.name,
            "--out": Path("batches") / report_path.name,
        }
        for flag, value in expected.items():
            recorded = options.get(flag)
            if flag in path_suffixes:
                suffix = path_suffixes[flag].parts
                _expect(
                    recorded is not None
                    and Path(recorded).parts[-len(suffix) :] == suffix,
                    f"command:{index}:{flag}",
                )
            else:
                _expect(recorded == value, f"command:{index}:{flag}")
        expected_settings = {
            "challenger": "seed504 E4",
            "current": "initializer",
            "games": 2,
            "simulations_per_side": 128,
            "c_puct": 1.25,
            "workers": 1,
            "seed": 488 + index,
            "seed_contract": "azlite_eval_seed_v2",
            "move_cap": 200,
            "random_opening_plies": 0,
        }
        _expect(
            command.get("settings") == expected_settings, f"command_settings:{index}"
        )
        for flag in (
            "--challenger-search-options-json",
            "--current-search-options-json",
        ):
            _expect(
                json.loads(options[flag])
                == {
                    k: v
                    for k, v in expected_options.items()
                    if k != "random_opening_plies"
                },
                f"command_options:{index}",
            )
        _expect(
            options.get("--suite-sha256") == notes.get("suite_sha256") == sha(opener),
            f"suite_association:{index}",
        )
        _expect(len(game_rows) == 2, f"batch_games:{index}")
        _expect(
            report.get("games_played") == 2 and report.get("games") == 2,
            f"report_games:{index}",
        )
        _expect(
            sum(g.get("winner") in ("challenger", "F") for g in game_rows)
            == report.get("wins"),
            f"report_wins:{index}",
        )
        _expect(
            sum(g.get("winner") == "draw" for g in game_rows) == report.get("draws"),
            f"report_draws:{index}",
        )
        _expect(
            sum(g.get("winner") in ("current", "initializer") for g in game_rows)
            == report.get("losses"),
            f"report_losses:{index}",
        )
        _expect(
            notes.get("challenger_simulations") == 128
            and notes.get("current_simulations") == 128
            and notes.get("workers_used") == 1
            and notes.get("base_seed") == 488 + index
            and notes.get("random_opening_plies") == 0,
            f"report_runtime:{index}",
        )
        reported_options = {
            k: v
            for k, v in expected_options.items()
            if k not in ("random_opening_plies", "root_prior_transform")
        }
        _expect(
            notes.get("search_options") == reported_options, f"report_options:{index}"
        )
        _expect(
            notes.get("exact_root_solve_threshold") == 16
            and notes.get("exact_root_objective") == "final_score_margin"
            and notes.get("exact_root_tie_rule")
            == "highest_legal_network_prior_then_lowest_move_index"
            and notes.get("exact_leaf_solve_mode") == "disabled",
            f"report_exact_root:{index}",
        )
        exact_root_identities = {
            "exact_root_solver": "native_kvtb_root_action_probe_v1",
            "exact_root_solver_max_solved_seeds": 16,
            "runtime_search_policy_mode": "puct_exact_root_hybrid",
            "exact_root_native_probe_sha256": runtime_identities[
                "model-artifact/runtime/kalah_v1_tablebase"
            ],
            "exact_root_tablebase_sha256": runtime_identities[
                "model-artifact/runtime/kalah_v1_21.kvtb"
            ],
            "runtime_search_policy_sha256": runtime_identities[
                "model-artifact/current/search_policy.json"
            ],
        }
        for key, value in exact_root_identities.items():
            recorded = notes.get(key)
            if recorded is None:
                recorded = notes.get("search_profile", {}).get(key)
            _expect(recorded == value, f"report_exact_identity:{index}:{key}")
        pair = [r for r in ledger if int(r["opening_index"]) == index]
        _expect(
            len(pair) == 2 and {int(r["F_player"]) for r in pair} == {0, 1},
            f"ledger_pair:{index}",
        )
        for row in pair:
            seat = int(row["F_player"])
            pos = int(row["batch_game_position"])
            game = game_rows[pos]
            _expect(
                row["batch_game_sha256"] == sha(game_path)
                and row["batch_report_sha256"] == sha(report_path)
                and row["batch_opening_sha256"] == sha(opener),
                f"ledger_binding:{index}:{seat}",
            )
            _expect(
                int(game["challenger_player"]) == seat
                and int(row["challenger_player"]) == seat,
                f"ledger_seat:{index}:{seat}",
            )
            _expect(
                row.get("trajectory") == game.get("trajectory")
                and int(row.get("opening_applied_prefix_length", -1))
                == int(game.get("opening_applied_prefix_length", -2))
                and row.get("opening_prefix_moves") == game.get("opening_prefix_moves"),
                f"ledger_game_fields:{index}:{seat}",
            )
            winner, margin, length = _replay_game(opening, game, seat)
            winner = {"F": "challenger", "current": "current"}.get(winner, winner)
            _expect(
                winner == row["winner"]
                and margin == int(row["margin"])
                and length == int(row["game_length"]),
                f"ledger_replay:{index}:{seat}",
            )
            points[index, seat] = (
                1.0 if winner == "challenger" else 0.5 if winner == "draw" else 0.0
            )
    wins = sum(r["winner"] == "challenger" for r in ledger)
    draws = sum(r["winner"] == "draw" for r in ledger)
    losses = sum(r["winner"] == "current" for r in ledger)
    score = (wins + 0.5 * draws) / 1024
    rng = np.random.default_rng(505)
    sampled = points.mean(axis=1)[rng.integers(0, 512, size=(20_000, 512))].mean(axis=1)
    lower = float(np.quantile(sampled, 0.05))
    interval = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    passed = lower > 0.5
    result = json.loads((out / "results.json").read_text())
    expected_result = {
        "games": 1024,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "bootstrap": {
            "replicates": 20_000,
            "seed": 505,
            "one_sided_lower_95": lower,
            "two_sided_95": interval,
        },
        "pass": passed,
        "promotion": False,
        "fresh_confirmation": False,
    }
    _expect(result == expected_result, "published_result_mismatch")
    return {
        "status": "verified_complete",
        **expected_result,
        "evidence_limitations": [
            "Per-game move cap and zero random-opening plies are recorded in commands; command inputs are validated. Exact per-game runtime seed internals are not embedded in trajectories beyond the report and command records."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
