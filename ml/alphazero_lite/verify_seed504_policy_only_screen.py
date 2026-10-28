"""Independent read-only reconstruction of the seed504 strength screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed504_policy_only_screen as protocol
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relocate_command(argv: list[str], root: Path) -> list[str]:
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


def _authenticate(root: Path, out: Path) -> dict[str, Any]:
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    suite_path = out / "openings.jsonl"
    if (
        registration.get("schema") != "seed504-policy-only-screen-v1"
        or registration.get("status") != "frozen_before_arena_execution"
        or freeze.get("registration_sha256") != _sha(registration_path)
        or freeze.get("suite_sha256") != _sha(suite_path)
        or registration["suite"].get("sha256") != _sha(suite_path)
        or registration["suite"].get("openings") != 1024
        or registration["suite"].get("games") != 2048
    ):
        raise ValueError("screen_freeze_mismatch")
    if len(_rows(suite_path)) != 1024:
        raise ValueError("opening_count_mismatch")
    if registration.get("assignments") != {
        "opponent_P": {
            "policy": "P-E4",
            "value": "initializer",
            "each_component_own_trunk": True,
        },
        "challenger_T": {
            "policy": "T-E4",
            "value": "initializer",
            "each_component_own_trunk": True,
        },
        "tensor_splicing": False,
    }:
        raise ValueError("component_assignment_mismatch")
    for name, component in registration["components"].items():
        checkpoint = root / component["checkpoint"]
        if _sha(checkpoint) != component["checkpoint_sha256"]:
            raise ValueError(f"component_checkpoint_mismatch:{name}")
        artifact = root / component["artifact"]
        for filename, digest in component["files"].items():
            if _sha(artifact / filename) != digest:
                raise ValueError(f"component_artifact_mismatch:{name}:{filename}")
    for rel, digest in registration["runtime_asset_sha256"].items():
        if _sha(root / rel) != digest:
            raise ValueError(f"runtime_identity_mismatch:{rel}")
    for rel, digest in registration["input_sha256"].items():
        if _sha(root / rel) != digest:
            raise ValueError(f"input_identity_mismatch:{rel}")
    for rel, digest in registration["source_sha256"].items():
        snapshot = out / "execution-source-snapshots" / f"{Path(rel).name}.txt"
        source = root / rel
        if (not source.is_file() or _sha(source) != digest) and (
            not snapshot.is_file() or _sha(snapshot) != digest
        ):
            raise ValueError(f"historical_source_identity_mismatch:{rel}")
    return registration


def bootstrap_lower_bound(
    opening_scores: list[float] | np.ndarray,
    *,
    resamples: int = 20_000,
    seed: int = 539,
    quantile: float = 0.05,
) -> tuple[float, np.ndarray]:
    """Resample opening clusters with replacement, retaining duplicate draws."""
    scores = np.asarray(opening_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), size=(resamples, len(scores)))
    draws = scores[indices].mean(axis=1)
    return float(np.quantile(draws, quantile, method="linear")), draws


def analyze_independently(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if len(ledger) != 2048:
        raise ValueError("analysis_requires_all_valid_pairs")
    values = {"challenger": 1.0, "draw": 0.5, "current": 0.0}
    grouped: dict[int, list[float]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(values[row["winner"]])
    if set(grouped) != set(range(1024)) or any(
        len(pair) != 2 for pair in grouped.values()
    ):
        raise ValueError("analysis_requires_all_valid_pairs")
    opening_scores = np.asarray([np.mean(grouped[i]) for i in range(1024)])
    lower, samples = bootstrap_lower_bound(opening_scores)
    wins = sum(row["winner"] == "challenger" for row in ledger)
    draws = sum(row["winner"] == "draw" for row in ledger)
    losses = sum(row["winner"] == "current" for row in ledger)
    if wins + draws + losses != len(ledger):
        raise ValueError("unknown_outcome")
    advance = advance_from_lower_bound(lower)
    return {
        "games": 2048,
        "openings": 1024,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / 2048,
        "bootstrap_resamples": 20_000,
        "bootstrap_seed": 539,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(samples, q, method="linear")) for q in (0.025, 0.975)
        ],
        "all_games_valid": True,
        "advance": advance,
        "advance_authorizes": "proposal for separate 1,200-simulation evaluation only"
        if advance
        else None,
        "deployment_or_promotion": False,
        "fresh_confirmation": False,
        "seed461_historical_overlap_limitation": "preserved; this reused suite is exploratory only",
    }


def advance_from_lower_bound(lower: float) -> bool:
    """The registered advancement boundary is strict, not inclusive."""
    return lower > 0.50


def _verify_pair(
    root: Path,
    out: Path,
    registration: dict[str, Any],
    opening: dict[str, Any],
    index: int,
) -> list[dict[str, Any]]:
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    games_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    if not all(path.is_file() for path in (opening_path, games_path, report_path)):
        raise ValueError(f"pair_incomplete:{index}")
    if (
        opening_path.read_bytes()
        != json.dumps(opening, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    ):
        raise ValueError(f"pair_opening_mismatch:{index}")
    games = _rows(games_path)
    report = json.loads(report_path.read_text())
    if len(games) != 2 or {int(game["challenger_player"]) for game in games} != {0, 1}:
        raise ValueError(f"pair_seat_mismatch:{index}")
    notes = report.get("notes", {})
    contract = registration["runtime"]
    expected_composition = {
        **registration["assignments"],
        "components": registration["components"],
        "runtime_asset_sha256": registration["runtime_asset_sha256"],
    }
    expected_options = {
        key: value
        for key, value in contract["search_options"].items()
        if key != "random_opening_plies" and value is not None
    }
    if (
        report.get("games_played") != 2
        or report.get("games") != 2
        or report.get("policy_ablation_composition") != expected_composition
        or notes.get("base_seed") != 539 + index
        or notes.get("seed") != 539 + index
        or notes.get("suite_sha256") != _sha(opening_path)
        or notes.get("challenger_simulations") != 128
        or notes.get("current_simulations") != 128
        or notes.get("workers_used") != 1
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("exact_root_solve_threshold") != 16
        or notes.get("exact_root_native_probe_sha256")
        != contract["native_probe_sha256"]
        or notes.get("exact_root_tablebase_sha256") != contract["tablebase_sha256"]
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("search_options") != expected_options
        or notes.get("search_profile", {}).get("search_options") != expected_options
    ):
        raise ValueError(f"pair_runtime_or_report_mismatch:{index}")
    outcomes: list[dict[str, Any]] = []
    for position, game in enumerate(games):
        if game.get("capped") or game.get("status") in {"capped", "incomplete"}:
            raise ValueError(f"capped_or_incomplete_game:{index}:{position}")
        if int(game["game_length"]) >= 200:
            raise ValueError(f"capped_or_incomplete_game:{index}:{position}")
        winner, margin, length = _replay_game(
            opening, game, int(game["challenger_player"])
        )
        if (
            winner != game.get("winner")
            or margin != int(game["margin"])
            or length != int(game["game_length"])
        ):
            raise ValueError(f"raw_outcome_mismatch:{index}:{position}")
        outcomes.append({"winner": winner, "margin": margin})
    report_counts = {
        "wins": sum(x["winner"] == "challenger" for x in outcomes),
        "draws": sum(x["winner"] == "draw" for x in outcomes),
        "losses": sum(x["winner"] == "current" for x in outcomes),
    }
    if any(report.get(key) != value for key, value in report_counts.items()):
        raise ValueError(f"report_outcome_disagreement:{index}")
    return [
        {
            "opening_index": index,
            "challenger_player": int(game["challenger_player"]),
            "winner": outcome["winner"],
            "batch_game_position": position,
            "batch_game_sha256": _sha(games_path),
            "batch_report_sha256": _sha(report_path),
            "batch_opening_sha256": _sha(opening_path),
        }
        for position, (game, outcome) in enumerate(zip(games, outcomes, strict=True))
    ]


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    registration = _authenticate(root, out)
    openings = _rows(out / "openings.jsonl")
    ledger = _rows(out / "outcomes.jsonl") if (out / "outcomes.jsonl").is_file() else []
    if len(ledger) % 2:
        raise ValueError("partial_ledger_pair")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    if any(
        index not in range(1024) or len(rows) != 2 for index, rows in grouped.items()
    ):
        raise ValueError("duplicate_or_invalid_ledger_pair")
    expected_pairs = {
        index: _verify_pair(root, out, registration, openings[index], index)
        for index in grouped
    }
    for index, expected in expected_pairs.items():
        if grouped[index] != expected:
            raise ValueError(f"ledger_report_disagreement:{index}")
    commands = (
        _rows(out / "commands.jsonl") if (out / "commands.jsonl").is_file() else []
    )
    by_index: dict[int, dict[str, Any]] = {}
    for item in commands:
        index = int(item["opening_index"])
        if index not in range(1024) or index in by_index:
            raise ValueError("command_index_duplicate_or_invalid")
        wanted = protocol.command(root, index, prepare_opening=False)
        got = _relocate_command(item.get("argv", []), root)
        if (
            got != wanted
            or item.get("registration_sha256") != _sha(out / "registration.json")
            or item.get("suite_sha256") != _sha(out / "openings.jsonl")
            or item.get("seed") != 539 + index
            or item.get("component_assignments") != registration["assignments"]
            or item.get("component_files")
            != {
                key: value["files"] for key, value in registration["components"].items()
            }
        ):
            raise ValueError(f"command_binding_mismatch:{index}")
        by_index[index] = item
    if not set(grouped).issubset(by_index):
        raise ValueError("committed_pair_missing_command")
    if len(grouped) != 1024 or len(ledger) != 2048:
        return {
            "status": "verified_incomplete",
            "validated_pairs": len(grouped),
            "validated_games": len(ledger),
            "advance": False,
        }
    actual = analyze_independently(ledger)
    if json.loads((out / "results.json").read_text()) != actual:
        raise ValueError("result_analysis_mismatch")
    return {
        "status": "verified_complete",
        "validated_pairs": len(grouped),
        "validated_games": len(ledger),
        "results": actual,
        "advance": actual["advance"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
