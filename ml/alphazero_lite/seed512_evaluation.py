"""Examined exploratory 1,200-simulation follow-up to seed511."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed511_fixed_composition_screen as seed511

ROOT = seed511.ROOT
OUT = Path("docs/data/seed512-fixed-composition-evaluation")
SEED = 512
SIMULATIONS = 1200
RESAMPLES = 20_000
SEED511_OUT = Path("docs/data/seed511-fixed-composition-screen-v4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def prepare(root: Path = ROOT) -> dict[str, Any]:
    """Bind the already-published roots, seed511 artifacts, and runtime prospectively."""
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed512_output_already_exists")
    source = root / SEED511_OUT
    published = _read_json(source / "registration.json")
    receipt = _read_json(source / "publication-receipt.json")
    openings_path = source / "openings.jsonl"
    openings = read_rows(openings_path)
    if len(openings) != 1024 or published["suite"]["sha256"] != sha(openings_path):
        raise ValueError("seed512_seed511_suite_identity")
    env = seed511.dependency_versions()
    expected_env = published["environment"]["pip_freeze"]
    if env != expected_env:
        raise ValueError("seed512_environment_dependency_mismatch")
    import torch

    if torch.__version__ != published["environment"]["torch"]:
        raise ValueError("seed512_torch_version_mismatch")
    runtime = dict(published["runtime"])
    runtime["simulations_per_side"] = SIMULATIONS
    registration = {
        "schema": "seed512-examined-exploratory-follow-up-v1",
        "status": "frozen_before_seed512_games",
        "interpretation": (
            "exploratory only; seed511 and its 128-simulation outcomes have already "
            "been examined; not fresh confirmation"
        ),
        "seed511_publication": {
            "directory": str(SEED511_OUT),
            "publication_receipt_sha256": sha(source / "publication-receipt.json"),
            "registration_sha256": sha(source / "registration.json"),
            "execution_freeze_sha256": sha(source / "execution-freeze.json"),
            "outcomes_sha256": sha(source / "outcomes.jsonl"),
            "results_sha256": sha(source / "results.json"),
            "results": _read_json(source / "results.json"),
            "examined_before_seed512": True,
        },
        "candidate": published["candidate"],
        "opponent": published["opponent"],
        "components": published["components"],
        "runtime_assets": published["runtime_assets"],
        "runtime": runtime,
        "suite": {
            "source": str(openings_path.relative_to(root)),
            "openings": 1024,
            "games": 2048,
            "sha256": sha(openings_path),
            "seed": 511,
            "identity": "exact seed511 published frozen opening roots; no regeneration or replacement",
        },
        "decision": {
            "primary_score": "(wins + 0.5 * draws) / 2048",
            "gate": "pass only if one-sided 95% lower bound > 0.50",
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": SEED,
            "promotion": False,
            "fresh_confirmation": False,
            "seed461_overlap_limitation_preserved": True,
            "seed505_separate_failed_gate_preserved": True,
            "passing_only_supports_later_deployment_and_evaluation_proposal": True,
        },
        "secondary": {
            "contrast": "matched per-opening seed512 minus seed511 score difference",
            "resamples": RESAMPLES,
            "seed": SEED,
            "resampling_unit": "opening; both seats kept together",
            "descriptive_only": True,
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": env,
            "pip_freeze_sha256": hashlib.sha256(canonical(env)).hexdigest(),
            "device": "CPU; torch.cuda.is_available()="
            + str(torch.cuda.is_available()),
        },
        "inputs_sha256": {
            "seed511_registration.json": sha(source / "registration.json"),
            "seed511_execution-freeze.json": sha(source / "execution-freeze.json"),
            "seed511_openings.jsonl": sha(openings_path),
            "seed511_outcomes.jsonl": sha(source / "outcomes.jsonl"),
            "seed511_results.json": sha(source / "results.json"),
        },
        "source_sha256": {
            "ml/alphazero_lite/seed512_evaluation.py": sha(
                root / "ml/alphazero_lite/seed512_evaluation.py"
            ),
            "ml/alphazero_lite/verify_seed512_evaluation.py": sha(
                root / "ml/alphazero_lite/verify_seed512_evaluation.py"
            ),
            "ml/alphazero_lite/test_seed512_evaluation.py": sha(
                root / "ml/alphazero_lite/test_seed512_evaluation.py"
            ),
        },
        "publication_file_count": receipt["file_count"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "openings.jsonl").write_bytes(openings_path.read_bytes())
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    freeze = {
        "status": "frozen_before_seed512_games",
        "registration_sha256": sha(out / "registration.json"),
        "suite_sha256": sha(out / "openings.jsonl"),
    }
    (out / "execution-freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    preflight(root)
    return freeze


def refreeze_after_preflight_edits(root: Path = ROOT) -> dict[str, Any]:
    """Supersede a pre-execution freeze if changes were made before any commands."""
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    freeze_path = out / "execution-freeze.json"
    if (out / "commands.jsonl").exists() or (out / "outcomes.jsonl").exists():
        raise ValueError("seed512_cannot_refreeze_after_start")
    registration = _read_json(reg_path)
    old_freeze = _read_json(freeze_path)
    if old_freeze.get("registration_sha256") != sha(reg_path):
        raise ValueError("seed512_existing_freeze_invalid")
    registration["superseded_pre_execution_freeze_sha256"] = sha(freeze_path)
    registration["source_sha256"] = {
        rel: sha(root / rel) for rel in registration["source_sha256"]
    }
    reg_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    freeze = {
        "status": "frozen_before_seed512_games",
        "registration_sha256": sha(reg_path),
        "suite_sha256": sha(out / "openings.jsonl"),
        "supersedes_pre_execution_freeze_sha256": sha(freeze_path),
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return preflight(root)


def command(root: Path, index: int) -> list[str]:
    """Construct the validated seed511 arena invocation, changing simulations only."""
    root = root.resolve()
    seed511_out = root / SEED511_OUT
    original_out = seed511.OUT
    try:
        seed511.OUT = OUT
        argv = seed511.command(
            root, index, read_rows(root / OUT / "openings.jsonl")[index]
        )
    finally:
        seed511.OUT = original_out
    for flag in ("--challenger-simulations", "--current-simulations"):
        argv[argv.index(flag) + 1] = str(SIMULATIONS)
    argv[argv.index("--seed") + 1] = str(511 + index)
    component_root = root / SEED511_OUT / "runtime-artifacts"
    for flag, relative in (
        ("--challenger", "E4"),
        ("--challenger-policy-artifact", "E4"),
        ("--challenger-value-artifact", "initializer"),
        ("--current", "initializer"),
    ):
        argv[argv.index(flag) + 1] = str(component_root / relative)
    # Resolve runtime sidecars from seed511's exact registered identities.
    reg = _read_json(seed511_out / "registration.json")
    for flag, key in (
        ("--exact-root-native-probe", "exact_root_native_probe"),
        ("--exact-root-tablebase", "exact_root_tablebase"),
    ):
        argv[argv.index(flag) + 1] = str(root / reg["runtime_assets"][key]["artifact"])
    return argv


def _points(winner: str) -> float:
    if winner == "challenger":
        return 1.0
    if winner == "draw":
        return 0.5
    if winner in ("current", "initializer"):
        return 0.0
    raise ValueError(f"seed512_unknown_winner:{winner}")


def resume_plan(
    root: Path, openings: list[dict[str, Any]], ledger: list[dict[str, Any]]
) -> list[int]:
    """Skip only verified committed pairs; preserve any uncommitted outcome-bearing files."""
    if len(ledger) % 2:
        raise ValueError("seed512_partial_ledger_pair")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    if any(i not in range(1024) or len(rows) != 2 for i, rows in grouped.items()):
        raise ValueError("seed512_ledger_pair_shape")
    for index, rows in grouped.items():
        if {int(row["challenger_player"]) for row in rows} != {0, 1}:
            raise ValueError(f"seed512_ledger_seats:{index}")
        verify_pair(root, index, openings[index], rows)
    batches = root / OUT / "batches"
    pending = []
    for index in range(1024):
        if index in grouped:
            continue
        paths = [
            batches / f"{kind}-{index:04d}.{ext}"
            for kind, ext in (
                ("games", "jsonl"),
                ("report", "json"),
                ("opening", "jsonl"),
            )
        ]
        if any(path.exists() for path in paths):
            raise ValueError(f"seed512_interrupted_pair_preserved:{index}")
        pending.append(index)
    return pending


def verify_pair(
    root: Path, index: int, opening: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Validate pair receipts, both seats, runtime settings, and replay legality."""
    out = root / OUT
    batches = out / "batches"
    game_path, report_path, opening_path = (
        batches / f"games-{index:04d}.jsonl",
        batches / f"report-{index:04d}.json",
        batches / f"opening-{index:04d}.jsonl",
    )
    games = read_rows(game_path)
    report = _read_json(report_path)
    notes = report.get("notes", {})
    if (
        opening_path.read_bytes() != seed511.canonical_json(opening) + b"\n"
        or len(games) != 2
        or report.get("games_played") != 2
    ):
        raise ValueError(f"seed512_pair_shape:{index}")
    if {int(g["challenger_player"]) for g in games} != {0, 1}:
        raise ValueError(f"seed512_pair_seats:{index}")
    assets = _read_json(root / SEED511_OUT / "registration.json")["runtime_assets"]
    frozen_runtime = _read_json(root / OUT / "registration.json")["runtime"]
    expected_options = seed511.base.arena_effective_search_options(
        _read_json(root / SEED511_OUT / "registration.json")
    )
    if (
        notes.get("challenger_simulations") != SIMULATIONS
        or notes.get("current_simulations") != SIMULATIONS
        or notes.get("workers_used") != 1
        or notes.get("workers_requested") != 1
        or notes.get("base_seed") != 511 + index
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("exact_root_solve_threshold") != 16
        or notes.get("exact_leaf_solve_mode") != "disabled"
        or notes.get("exact_root_objective") != "final_score_margin"
        or notes.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("exact_root_native_probe_sha256")
        != assets["exact_root_native_probe"]["sha256"]
        or notes.get("exact_root_tablebase_sha256")
        != assets["exact_root_tablebase"]["sha256"]
        or notes.get("random_opening_plies") != 0
        or notes.get("search_options") != expected_options
        or notes.get("search_profile", {}).get("search_options") != expected_options
        or report.get("seed512_followup")
        != {
            "examined_exploratory_follow_up": True,
            "seed511_128_simulation_outcomes_previously_examined": True,
            "policy_source": "seed504 E4 with own trunk",
            "value_source": "initializer with own trunk",
            "opponent": "complete frozen initializer",
            "simulations_per_side": SIMULATIONS,
            "runtime_contract": frozen_runtime,
        }
    ):
        raise ValueError(f"seed512_pair_runtime:{index}")
    if len(rows) != 2:
        raise ValueError(f"seed512_pair_ledger:{index}")
    from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

    for row in rows:
        pos = int(row["batch_game_position"])
        if (
            row["batch_game_sha256"] != sha(game_path)
            or row["batch_report_sha256"] != sha(report_path)
            or row["batch_opening_sha256"] != sha(opening_path)
        ):
            raise ValueError(f"seed512_pair_receipt:{index}:{pos}")
        if int(games[pos]["challenger_player"]) != int(row["challenger_player"]):
            raise ValueError(f"seed512_pair_seat_receipt:{index}:{pos}")
        winner, _, _ = _replay_game(opening, games[pos], int(row["challenger_player"]))
        if winner != row["winner"]:
            raise ValueError(f"seed512_pair_replay:{index}:{pos}")


def analyze(root: Path, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    point_by_opening: dict[int, list[float]] = {i: [] for i in range(1024)}
    for row in ledger:
        point_by_opening[int(row["opening_index"])].append(_points(row["winner"]))
    if len(ledger) != 2048 or any(len(v) != 2 for v in point_by_opening.values()):
        raise ValueError("seed512_analysis_pairing")
    cluster = np.array([np.mean(point_by_opening[i]) for i in range(1024)])
    rng = np.random.default_rng(SEED)
    resample_indices = rng.integers(0, 1024, (RESAMPLES, 1024))
    draws = cluster[resample_indices].mean(axis=1)
    wins = sum(row["winner"] == "challenger" for row in ledger)
    ties = sum(row["winner"] == "draw" for row in ledger)
    prior = read_rows(root / SEED511_OUT / "outcomes.jsonl")
    previous: dict[int, list[float]] = {i: [] for i in range(1024)}
    for row in prior:
        previous[int(row["opening_index"])].append(_points(row["winner"]))
    delta = np.array([cluster[i] - np.mean(previous[i]) for i in range(1024)])
    contrast_draws = delta[resample_indices].mean(axis=1)
    lower = float(np.quantile(draws, 0.05))
    return {
        "games": len(ledger),
        "wins": wins,
        "draws": ties,
        "losses": len(ledger) - wins - ties,
        "score": (wins + 0.5 * ties) / 2048,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "pass": lower > 0.50,
        "promotion": False,
        "fresh_confirmation": False,
        "interpretation": "examined exploratory follow-up only; seed511 and its 128-simulation outcomes were already examined",
        "seed511_matched_contrast": {
            "mean_1200_minus_128": float(delta.mean()),
            "descriptive_two_sided_95": [
                float(np.quantile(contrast_draws, 0.025)),
                float(np.quantile(contrast_draws, 0.975)),
            ],
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": SEED,
            "resampling_unit": "paired opening, keeping both seats together",
            "primary_gate_substitute": False,
        },
    }


def preflight(root: Path = ROOT, *, allow_started: bool = False) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path, freeze_path, suite_path = (
        out / "registration.json",
        out / "execution-freeze.json",
        out / "openings.jsonl",
    )
    reg, freeze = _read_json(reg_path), _read_json(freeze_path)
    if (
        reg["status"] != "frozen_before_seed512_games"
        or freeze["registration_sha256"] != sha(reg_path)
        or freeze["suite_sha256"] != sha(suite_path)
        or sha(suite_path) != reg["suite"]["sha256"]
    ):
        raise ValueError("seed512_freeze_mismatch")
    source = root / SEED511_OUT
    for name, digest in reg["inputs_sha256"].items():
        if sha(source / name.removeprefix("seed511_")) != digest:
            raise ValueError(f"seed512_input_changed:{name}")
    if seed511.dependency_versions() != reg["environment"]["pip_freeze"]:
        raise ValueError("seed512_environment_changed")
    for rel, digest in reg["source_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"seed512_source_changed:{rel}")
    if (
        len(read_rows(suite_path)) != 1024
        or suite_path.read_bytes() != (source / "openings.jsonl").read_bytes()
    ):
        raise ValueError("seed512_suite_changed")
    if not allow_started and (
        (out / "commands.jsonl").exists() or (out / "outcomes.jsonl").exists()
    ):
        raise ValueError("seed512_preflight_after_start")
    return {
        "status": "preflight_passed",
        "registration_sha256": freeze["registration_sha256"],
        "suite_sha256": freeze["suite_sha256"],
        "openings": 1024,
    }


def run(
    root: Path = ROOT, *, execute: bool = False, max_pairs: int | None = None
) -> dict[str, Any]:
    if not execute:
        raise ValueError("seed512_execution_requires_explicit_flag")
    root = root.resolve()
    preflight(root, allow_started=True)
    out = root / OUT
    openings = read_rows(out / "openings.jsonl")
    ledger_path = out / "outcomes.jsonl"
    ledger = read_rows(ledger_path) if ledger_path.exists() else []
    pending = resume_plan(root, openings, ledger)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("seed512_invalid_pair_limit")
        pending = pending[:max_pairs]
    for index in pending:
        argv = command(root, index)
        opening_path = out / "batches" / f"opening-{index:04d}.jsonl"
        opening_path.parent.mkdir(parents=True, exist_ok=True)
        opening_path.write_bytes(seed511.canonical_json(openings[index]) + b"\n")
        started = time.monotonic()
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "opening_index": index,
                        "argv": argv,
                        "component_assignments": {
                            "policy": "seed504 E4, own trunk",
                            "value": "initializer, own trunk",
                            "opponent": "complete frozen initializer",
                        },
                        "registration_sha256": sha(out / "registration.json"),
                        "suite_sha256": sha(out / "openings.jsonl"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        seed511_registration = _read_json(root / SEED511_OUT / "registration.json")
        tablebase_pointer = seed511.materialize_bound_tablebase(
            root, seed511_registration
        )
        try:
            subprocess.run(argv, cwd=root, check=True)
        finally:
            seed511.restore_bound_tablebase(
                root, seed511_registration, tablebase_pointer
            )
        elapsed = time.monotonic() - started
        game_path, report_path = (
            out / "batches" / f"games-{index:04d}.jsonl",
            out / "batches" / f"report-{index:04d}.json",
        )
        report = _read_json(report_path)
        report["seed512_elapsed_seconds"] = elapsed
        report["seed512_followup"] = {
            "examined_exploratory_follow_up": True,
            "seed511_128_simulation_outcomes_previously_examined": True,
            "policy_source": "seed504 E4 with own trunk",
            "value_source": "initializer with own trunk",
            "opponent": "complete frozen initializer",
            "simulations_per_side": SIMULATIONS,
            "runtime_contract": _read_json(out / "registration.json")["runtime"],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        games = read_rows(game_path)
        batch_rows = [
            {
                "opening_index": index,
                "challenger_player": int(g["challenger_player"]),
                "winner": g["winner"],
                "batch_game_position": pos,
                "batch_game_sha256": sha(game_path),
                "batch_report_sha256": sha(report_path),
                "batch_opening_sha256": sha(opening_path),
            }
            for pos, g in enumerate(games)
        ]
        if len(batch_rows) != 2:
            raise ValueError(f"seed512_incomplete_pair:{index}")
        verify_pair(root, index, openings[index], batch_rows)
        ledger.extend(batch_rows)
        tmp = ledger_path.with_suffix(".jsonl.pending")
        tmp.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger)
        )
        tmp.replace(ledger_path)
        if index == 0:
            (out / "first-pair-timing.json").write_text(
                json.dumps(
                    {
                        "opening_index": 0,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed * 1023,
                        "games_complete": 2,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    if len(ledger) < 2048:
        return {
            "status": "in_progress",
            "committed_games": len(ledger),
            "completed_openings": len(ledger) // 2,
            "next_opening": next(
                (
                    i
                    for i in range(1024)
                    if i not in {int(r["opening_index"]) for r in ledger}
                ),
                None,
            ),
        }
    result = analyze(root, ledger)
    (out / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--first-pair", action="store_true")
    parser.add_argument("--refreeze-after-preflight-edits", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.refreeze_after_preflight_edits:
        result = refreeze_after_preflight_edits()
    elif args.first_pair:
        result = run(execute=True, max_pairs=1)
    elif args.execute:
        result = run(execute=True)
    else:
        parser.error("choose --prepare or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
