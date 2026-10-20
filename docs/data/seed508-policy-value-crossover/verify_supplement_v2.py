"""Independent seed508 v2 arithmetic and report-contract audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.verify_seed508_policy_value_crossover import verify_complete_pair
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

ARMS = ("P", "V")
OPENINGS = 512
RESAMPLES = 20_000


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _points(winner: str) -> float:
    return 1.0 if winner == "challenger" else 0.5 if winner == "draw" else 0.0


def _flag(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"command_flag:{name}") from exc


def recompute(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute published seed508 fields from replay-verified outcome rows."""
    by_arm = {
        arm: {
            i: [
                row for row in ledger
                if row["arm"] == arm and int(row["opening_index"]) == i
            ]
            for i in range(OPENINGS)
        }
        for arm in ARMS
    }
    per_arm: dict[str, np.ndarray] = {}
    standalone: dict[str, dict[str, Any]] = {}
    for arm, offset in (("P", 1), ("V", 2)):
        per_arm[arm] = np.array(
            [sum(_points(row["winner"]) for row in by_arm[arm][i]) / 2
             for i in range(OPENINGS)]
        )
        rng = np.random.default_rng(508 + offset)
        sampled = per_arm[arm][rng.integers(0, OPENINGS, (RESAMPLES, OPENINGS))].mean(axis=1)
        wins = sum(row["arm"] == arm and row["winner"] == "challenger" for row in ledger)
        draws = sum(row["arm"] == arm and row["winner"] == "draw" for row in ledger)
        standalone[arm] = {
            "games": len(ledger) // 2,
            "wins": wins,
            "draws": draws,
            "losses": len(ledger) // 2 - wins - draws,
            "score": (wins + 0.5 * draws) / (len(ledger) // 2),
            "descriptive_two_sided_95": [
                float(np.quantile(sampled, 0.025)),
                float(np.quantile(sampled, 0.975)),
            ],
        }
    difference = per_arm["P"] - per_arm["V"]
    bootstrap = difference[np.random.default_rng(508).integers(
        0, OPENINGS, (RESAMPLES, OPENINGS)
    )].mean(axis=1)
    interval = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    return {
        "standalone": standalone,
        "primary": {
            "estimand": "per-opening P score minus V score, each score averages two seat games",
            "mean_difference": float(difference.mean()),
            "bootstrap_resamples": RESAMPLES,
            "seed": 508,
            "two_sided_95": interval,
            "interpretation": "P supported" if interval[0] > 0 else "V supported" if interval[1] < 0 else "inconclusive",
        },
    }


def verify(root: Path, out: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    out = (out or root / "docs/data/seed508-policy-value-crossover").resolve()
    registration = json.loads((out / "registration.json").read_text())
    for name, component in registration["components"].items():
        artifact_dir = out / component["artifact"]
        for filename, digest in component["files"].items():
            artifact_path = artifact_dir / filename
            if (
                not artifact_path.is_file()
                or hashlib.sha256(artifact_path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"registered_component_artifact:{name}:{filename}")
    commands = _rows(out / "commands.jsonl")
    ledger = _rows(out / "outcomes.jsonl")
    grouped = {(arm, i): [] for arm in ARMS for i in range(OPENINGS)}
    for row in ledger:
        grouped[(row["arm"], int(row["opening_index"]))].append(row)
    openings = _rows(out / "openings.jsonl")
    for arm in ARMS:
        for i, opening in enumerate(openings):
            verify_complete_pair(root, arm, i, opening, ledger)

    search = registration["runtime"]["search_options"]
    components = registration["components"]
    for record in commands:
        arm, i = record["arm"], int(record["opening_index"])
        argv = record["argv"]
        assignment = registration["arms"][arm]
        argument_components = {
            "--current": "initializer",
            "--challenger": assignment["policy_source"],
            "--challenger-policy-artifact": assignment["policy_source"],
            "--challenger-value-artifact": assignment["value_source"],
        }
        for flag, source in argument_components.items():
            actual = Path(_flag(argv, flag)).parts
            expected = Path(components[source]["artifact"]).parts
            if actual[-len(expected):] != expected:
                raise ValueError(f"command_component_binding:{arm}:{i}:{flag}")
        for field, source in (("policy", assignment["policy_source"]), ("value", assignment["value_source"])):
            artifact = Path(record.get("component_artifacts", {}).get(field, ""))
            expected = Path(components[source]["artifact"])
            if artifact.parts[-len(expected.parts):] != expected.parts:
                raise ValueError(f"command_component_identity:{arm}:{i}:{field}")
        report = json.loads((out / "batches" / f"{arm}-report-{i:04d}.json").read_text())
        profile = report["notes"]["search_profile"]
        notes = report["notes"]
        for note, flag in (("current_path", "--current"), ("challenger_path", "--challenger")):
            if Path(notes[note]).parts[-2:] != Path(_flag(argv, flag)).parts[-2:]:
                raise ValueError(f"report_opponent_binding:{arm}:{i}:{note}")
        resolved_search = {
            key: value for key, value in search.items()
            if key != "random_opening_plies" and value is not None
        }
        if profile.get("search_options") != resolved_search:
            raise ValueError(f"report_search_options:{arm}:{i}")
        for option in ("--challenger-search-options-json", "--current-search-options-json"):
            argv = record["argv"]
            actual = json.loads(argv[argv.index(option) + 1])
            if actual != {key: value for key, value in search.items() if key != "random_opening_plies"}:
                raise ValueError(f"resolved_search_options:{arm}:{i}:{option}")
        if profile.get("exact_leaf_solve_mode") != "disabled":
            raise ValueError(f"report_exact_leaf_mode:{arm}:{i}")
        composition = report.get("seed508_composition", {})
        for channel, source in (("policy", registration["arms"][arm]["policy_source"]), ("value", registration["arms"][arm]["value_source"])):
            expected = Path(components[source]["artifact"]).parts[-2:]
            if Path(composition.get("component_artifacts", {}).get(channel, "")).parts[-2:] != expected:
                raise ValueError(f"report_sidecar_identity:{arm}:{i}:{channel}")
        if composition.get("component_file_sha256") != components:
            raise ValueError(f"report_sidecar_identity:{arm}:{i}:manifest")

    for arm in ARMS:
        for i, opening in enumerate(openings):
            games_path = out / "batches" / f"{arm}-games-{i:04d}.jsonl"
            report_path = out / "batches" / f"{arm}-report-{i:04d}.json"
            opening_path = out / "batches" / f"{arm}-opening-{i:04d}.jsonl"
            games = _rows(games_path)
            pair = grouped[(arm, i)]
            if len(games) != 2 or len(pair) != 2:
                raise ValueError(f"copied_pair_shape:{arm}:{i}")
            for result in pair:
                pos = int(result["batch_game_position"])
                game = games[pos]
                if (
                    hashlib.sha256(games_path.read_bytes()).hexdigest()
                    != result["batch_game_sha256"]
                    or hashlib.sha256(report_path.read_bytes()).hexdigest()
                    != result["batch_report_sha256"]
                    or hashlib.sha256(opening_path.read_bytes()).hexdigest()
                    != result["batch_opening_sha256"]
                ):
                    raise ValueError(f"copied_pair_receipt:{arm}:{i}:{pos}")
                winner, _, _ = _replay_game(opening, game, int(result["challenger_player"]))
                if winner != result["winner"]:
                    raise ValueError(f"copied_pair_replay:{arm}:{i}:{pos}")

    actual = recompute(ledger)
    published = json.loads((out / "results.json").read_text())
    if published.get("standalone") != actual["standalone"]:
        raise ValueError("published_standalone_arithmetic_mismatch")
    if published.get("primary") != actual["primary"]:
        raise ValueError("published_primary_arithmetic_mismatch")
    return {"status": "verified", "commands": len(commands), "outcomes": len(ledger), "recomputed": actual}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))
