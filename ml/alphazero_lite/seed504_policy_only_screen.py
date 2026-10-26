"""Frozen, resumable seed511-suite screen for the seed504 policy-only ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed488_strength_evaluation as arena_protocol
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed504-policy-only-ablation/screen")
SEED511 = Path("docs/data/seed511-fixed-composition-screen-v4")
SEED504 = Path("docs/data/seed504")
ABLA = Path("docs/data/seed504-policy-only-ablation")
OPENINGS = 1024
GAMES = 2048
BASE_SEED = 539
BOOTSTRAP_SEED = 539
RESAMPLES = 20_000
SIMULATIONS = 128
ARTIFACT_FILES = ("model.npz", "weights.json", "metadata.json", "search_policy.json")
RUNTIME_FILES = (
    "model-artifact/runtime/kalah_v1_tablebase",
    "model-artifact/runtime/kalah_v1_21.kvtb",
)
SOURCE_FILES = (
    "ml/alphazero_lite/seed504_policy_only_screen.py",
    "ml/alphazero_lite/verify_seed504_policy_only_screen.py",
    "ml/alphazero_lite/test_seed504_policy_only_screen.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/seed488_strength_evaluation.py",
    "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
    "ml/alphazero_lite/seed511_fixed_composition_screen.py",
    "ml/alphazero_lite/verify_seed511_fixed_composition_screen.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def atomic_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    pending.replace(path)


def export_candidate(root: Path, destination: Path) -> None:
    """Build T's full runtime artifact from its saved model; never splice tensors."""
    source = root / SEED511 / "runtime-artifacts/E4"
    shutil.copytree(source, destination)
    checkpoint = root / ABLA / "seed504_policy_only_epoch4.npz"
    shutil.copyfile(checkpoint, destination / "model.npz")
    with np.load(checkpoint, allow_pickle=False) as archive:
        weights = {name: archive[name].tolist() for name in archive.files}
    (destination / "weights.json").write_text(
        json.dumps(weights, sort_keys=True, separators=(",", ":")) + "\n"
    )
    metadata = json.loads((destination / "metadata.json").read_text())
    metadata["version"] = "seed504-policy-only-T-E4"
    metadata["artifacts"]["weights_sha256"] = sha(destination / "model.npz")
    metadata["artifacts"]["weights_json_sha256"] = sha(destination / "weights.json")
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def component_files(path: Path) -> dict[str, str]:
    return {name: sha(path / name) for name in ARTIFACT_FILES}


def prepare(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists():
        raise ValueError("screen_publication_exists")
    dev_verification = json.loads(
        subprocess.run(
            [sys.executable, "-m", "ml.alphazero_lite.verify_seed504_policy_only"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if dev_verification.get("status") != "verified_development_incomplete_arena":
        raise ValueError("training_development_verification_required")
    seed511_registration = json.loads(
        (root / SEED511 / "registration.json").read_text()
    )
    seed511_freeze = json.loads((root / SEED511 / "execution-freeze.json").read_text())
    suite_path = root / SEED511 / "openings.jsonl"
    if (
        sha(root / SEED511 / "registration.json")
        != seed511_freeze["registration_sha256"]
        or sha(suite_path) != seed511_freeze["suite_sha256"]
    ):
        raise ValueError("seed511_suite_binding_mismatch")
    opening_rows = read_rows(suite_path)
    if len(opening_rows) != OPENINGS:
        raise ValueError("seed511_suite_count")
    out.mkdir(parents=True)
    (out / "batches").mkdir()
    shutil.copyfile(suite_path, out / "openings.jsonl")
    shutil.copytree(
        root / SEED511 / "runtime-artifacts/initializer",
        out / "runtime-artifacts/initializer",
    )
    shutil.copytree(
        root / SEED511 / "runtime-artifacts/E4", out / "runtime-artifacts/P-E4"
    )
    export_candidate(root, out / "runtime-artifacts/T-E4")
    components = {
        key: {
            "artifact": str((Path(OUT) / "runtime-artifacts" / directory).as_posix()),
            "files": component_files(out / "runtime-artifacts" / directory),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(root / checkpoint),
            "trunk": "complete artifact; not tensor-spliced",
        }
        for key, directory, checkpoint in (
            ("P_policy", "P-E4", SEED504 / "seed504_epoch4.npz"),
            ("T_policy", "T-E4", ABLA / "seed504_policy_only_epoch4.npz"),
            (
                "shared_value",
                "initializer",
                Path("docs/data/seed508-policy-value-crossover/initializer.npz"),
            ),
        )
    }
    components["shared_value"]["checkpoint_sha256"] = sha(
        root / components["shared_value"]["checkpoint"]
    )
    sources = {rel: sha(root / rel) for rel in SOURCE_FILES}
    runtime = {rel: sha(root / rel) for rel in RUNTIME_FILES}
    inputs = {
        str(path): sha(root / path)
        for path in (
            SEED511 / "registration.json",
            SEED511 / "execution-freeze.json",
            SEED511 / "openings.jsonl",
        )
    }
    import torch

    dependencies = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    registration = {
        "schema": "seed504-policy-only-screen-v1",
        "status": "frozen_before_arena_execution",
        "interpretation": "exploratory reuse of seed511 suite; preserves disclosed seed461 historical-overlap limitation; not fresh confirmation",
        "suite": {
            "source": str(SEED511 / "openings.jsonl"),
            "sha256": sha(out / "openings.jsonl"),
            "openings": OPENINGS,
            "games": GAMES,
            "reuse": "exact authenticated seed511 opening suite; no replacement or outcome-based filtering",
            "seed511_registration_sha256": sha(root / SEED511 / "registration.json"),
            "seed511_freeze_sha256": sha(root / SEED511 / "execution-freeze.json"),
            "seed461_limitation": "historical overlap limitation retained; no provenance recovery or fresh-confirmation claim",
        },
        "components": components,
        "assignments": {
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
        },
        "runtime": {
            "games_per_opening": 2,
            "seat_assignment": "one game for each challenger_player in {0,1}",
            "simulations_per_side": SIMULATIONS,
            "workers": 1,
            "c_puct": 1.25,
            "seed_contract": "azlite_eval_seed_v2",
            "base_seed": BASE_SEED,
            "game_seed": "base_seed + opening_index",
            "move_cap": 200,
            "capped_game_policy": "never count as ordinary draw; incomplete pairs stay uncommitted and require explicit adjudication",
            "exact_root_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solve": "disabled",
            "search_options": seed511_registration["runtime"]["search_options"],
            "native_probe_sha256": runtime[RUNTIME_FILES[0]],
            "tablebase_sha256": runtime[RUNTIME_FILES[1]],
        },
        "analysis": {
            "score": "(challenger wins + 0.5 * draws) / completed valid games",
            "cluster": "opening; mean two swapped-seat game scores, preserving sampled multiplicity",
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "one_sided_lower_quantile": 0.05,
            "two_sided_quantiles": [0.025, 0.975],
            "quantile_method": "linear",
            "advance_only_if": "all 2,048 games validate and lower bound > 0.50",
            "advance_authorizes": "proposal for a separate 1,200-simulation evaluation only; not deployment or promotion",
        },
        "source_sha256": sources,
        "input_sha256": inputs,
        "runtime_asset_sha256": runtime,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": dependencies,
            "launch_resource_contract": {
                "location": "host, outside API container and CI",
                "systemd_scope": ["CPUQuota=25%", "CPUWeight=1"],
                "affinity_cpu": 31,
                "nice": 19,
                "api_cgroup_observed_cpu_max": "200000 100000",
                "api_cgroup_cpuset": "0-31",
            },
        },
    }
    atomic_json(out / "registration.json", registration)
    atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_arena_execution",
            "registration_sha256": sha(out / "registration.json"),
            "suite_sha256": sha(out / "openings.jsonl"),
        },
    )
    return registration


def preflight(root: Path = ROOT, *, allow_started: bool = True) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if (
        registration.get("status") != "frozen_before_arena_execution"
        or freeze.get("registration_sha256") != sha(registration_path)
        or freeze.get("suite_sha256") != sha(out / "openings.jsonl")
        or registration["suite"]["sha256"] != sha(out / "openings.jsonl")
    ):
        raise ValueError("screen_freeze_mismatch")
    for rel, digest in registration["source_sha256"].items():
        source = root / rel
        snapshot = out / "execution-source-snapshots" / f"{Path(rel).name}.txt"
        if (not source.is_file() or sha(source) != digest) and (
            not snapshot.is_file() or sha(snapshot) != digest
        ):
            raise ValueError(f"frozen_source_changed:{rel}")
    for rel, digest in registration["runtime_asset_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_runtime_changed:{rel}")
    for rel, digest in registration["input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_input_changed:{rel}")
    for item in registration["components"].values():
        if sha(root / item["checkpoint"]) != item["checkpoint_sha256"]:
            raise ValueError("frozen_component_checkpoint_changed")
        path = root / item["artifact"]
        if component_files(path) != item["files"]:
            raise ValueError("frozen_component_artifact_changed")
    ledger = out / "outcomes.jsonl"
    if not allow_started and (ledger.exists() or (out / "commands.jsonl").exists()):
        raise ValueError("arena_already_started")
    openings = read_rows(out / "openings.jsonl")
    if len(openings) != OPENINGS:
        raise ValueError("opening_count_mismatch")
    return registration


def command(root: Path, index: int, *, prepare_opening: bool = True) -> list[str]:
    root = root.resolve()
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    opening = read_rows(out / "openings.jsonl")[index]
    expected = canonical(opening) + b"\n"
    if opening_path.exists() and opening_path.read_bytes() != expected:
        raise ValueError("opening_batch_identity_conflict")
    if prepare_opening and not opening_path.exists():
        opening_path.write_bytes(expected)
    opening_digest = (
        sha(opening_path)
        if opening_path.exists()
        else hashlib.sha256(expected).hexdigest()
    )
    artifacts = out / "runtime-artifacts"
    options = json.dumps(
        arena_protocol.arena_search_overrides(registration), sort_keys=True
    )
    argv = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(artifacts / "T-E4"),
        "--current",
        str(artifacts / "P-E4"),
        "--challenger-policy-artifact",
        str(artifacts / "T-E4"),
        "--challenger-value-artifact",
        str(artifacts / "initializer"),
        "--current-policy-artifact",
        str(artifacts / "P-E4"),
        "--current-value-artifact",
        str(artifacts / "initializer"),
        "--games",
        "2",
        "--games-per-opening",
        "2",
        "--workers",
        "1",
        "--challenger-simulations",
        str(SIMULATIONS),
        "--current-simulations",
        str(SIMULATIONS),
        "--c-puct",
        "1.25",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--challenger-search-options-json",
        options,
        "--current-search-options-json",
        options,
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--exact-root-native-probe",
        str(root / RUNTIME_FILES[0]),
        "--exact-root-tablebase",
        str(root / RUNTIME_FILES[1]),
        "--seed",
        str(BASE_SEED + index),
        "--opening-prefixes-jsonl",
        str(opening_path),
        "--suite-sha256",
        opening_digest,
        "--game-jsonl",
        str(batch / f"games-{index:04d}.jsonl"),
        "--out",
        str(batch / f"report-{index:04d}.json"),
    ]
    return argv


def _pair_rows(root: Path, index: int, opening: dict[str, Any]) -> list[dict[str, Any]]:
    out = root / OUT
    batch = out / "batches"
    game_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    if not all(path.is_file() for path in (game_path, report_path, opening_path)):
        raise ValueError(f"pair_incomplete:{index}")
    games = read_rows(game_path)
    report = json.loads(report_path.read_text())
    if (
        opening_path.read_bytes() != canonical(opening) + b"\n"
        or len(games) != 2
        or {int(row["challenger_player"]) for row in games} != {0, 1}
        or report.get("games_played") != 2
    ):
        raise ValueError(f"pair_shape:{index}")
    notes = report.get("notes", {})
    if (
        notes.get("challenger_simulations") != SIMULATIONS
        or notes.get("current_simulations") != SIMULATIONS
        or notes.get("workers_used") != 1
        or notes.get("base_seed") != BASE_SEED + index
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("exact_root_solve_threshold") != 16
        or notes.get("exact_leaf_solve_mode") != "disabled"
        or notes.get("suite_sha256") != sha(opening_path)
        or notes.get("exact_root_objective") != "final_score_margin"
        or notes.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("exact_root_objective") != "final_score_margin"
        or notes.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("exact_root_native_probe_sha256")
        != json.loads((out / "registration.json").read_text())["runtime"][
            "native_probe_sha256"
        ]
        or notes.get("exact_root_tablebase_sha256")
        != json.loads((out / "registration.json").read_text())["runtime"][
            "tablebase_sha256"
        ]
    ):
        raise ValueError(f"pair_runtime_contract:{index}")
    expected_options = arena_protocol.arena_effective_search_options(
        json.loads((out / "registration.json").read_text())
    )
    if (
        notes.get("search_options") != expected_options
        or notes.get("search_profile", {}).get("search_options") != expected_options
    ):
        raise ValueError(f"pair_search_options:{index}")
    result = []
    for position, game in enumerate(games):
        if (
            game.get("capped")
            or game.get("status") in {"capped", "incomplete"}
            or int(game.get("game_length", -1)) >= 200
        ):
            raise ValueError(f"capped_or_incomplete_game:{index}:{position}")
        winner, _, _ = _replay_game(opening, game, int(game["challenger_player"]))
        if winner != game.get("winner"):
            raise ValueError(f"winner_replay_mismatch:{index}:{position}")
        result.append(
            {
                "opening_index": index,
                "challenger_player": int(game["challenger_player"]),
                "winner": winner,
                "batch_game_position": position,
                "batch_game_sha256": sha(game_path),
                "batch_report_sha256": sha(report_path),
                "batch_opening_sha256": sha(opening_path),
            }
        )
    return result


def _verify_committed(
    root: Path, openings: list[dict[str, Any]], ledger: list[dict[str, Any]]
) -> set[int]:
    if len(ledger) % 2:
        raise ValueError("partial_ledger_pair")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    for index, rows in grouped.items():
        if index not in range(OPENINGS) or len(rows) != 2:
            raise ValueError("duplicate_or_invalid_ledger_pair")
        expected = _pair_rows(root, index, openings[index])
        if rows != expected:
            raise ValueError(f"ledger_receipt_mismatch:{index}")
    return set(grouped)


def resume_plan(
    root: Path, openings: list[dict[str, Any]], ledger: list[dict[str, Any]]
) -> tuple[list[int], list[int]]:
    committed = _verify_committed(root, openings, ledger)
    out = root / OUT
    recover: list[int] = []
    pending: list[int] = []
    for index in range(OPENINGS):
        if index in committed:
            continue
        batch = out / "batches"
        evidence = [
            batch / f"games-{index:04d}.jsonl",
            batch / f"report-{index:04d}.json",
            batch / f"opening-{index:04d}.jsonl",
        ]
        if any(path.exists() for path in evidence):
            # Outcome-bearing complete pairs are recovered. Partial pairs are
            # retained and rejected rather than silently rerun or counted.
            recovered = _pair_rows(root, index, openings[index])
            recover.extend([index] if len(recovered) == 2 else [])
        else:
            pending.append(index)
    return recover, pending


def analyze(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    points = {"challenger": 1.0, "draw": 0.5, "current": 0.0}
    grouped: dict[int, list[float]] = {}
    for row in ledger:
        grouped.setdefault(int(row["opening_index"]), []).append(points[row["winner"]])
    if (
        len(ledger) != GAMES
        or len(grouped) != OPENINGS
        or any(len(x) != 2 for x in grouped.values())
    ):
        raise ValueError("analysis_requires_all_valid_pairs")
    opening_scores = np.asarray([np.mean(grouped[i]) for i in range(OPENINGS)])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = opening_scores[rng.integers(0, OPENINGS, (RESAMPLES, OPENINGS))].mean(
        axis=1
    )
    wins = sum(row["winner"] == "challenger" for row in ledger)
    draws = sum(row["winner"] == "draw" for row in ledger)
    score = (wins + 0.5 * draws) / GAMES
    lower = float(np.quantile(samples, 0.05, method="linear"))
    return {
        "games": GAMES,
        "openings": OPENINGS,
        "wins": wins,
        "draws": draws,
        "losses": GAMES - wins - draws,
        "score": score,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(samples, q, method="linear")) for q in (0.025, 0.975)
        ],
        "all_games_valid": True,
        "advance": lower > 0.50,
        "advance_authorizes": "proposal for separate 1,200-simulation evaluation only"
        if lower > 0.50
        else None,
        "deployment_or_promotion": False,
        "fresh_confirmation": False,
        "seed461_historical_overlap_limitation": "preserved; this reused suite is exploratory only",
    }


def run(root: Path = ROOT, *, execute: bool = False) -> dict[str, Any]:
    if not execute:
        raise ValueError("execution_requires_explicit_--execute")
    root = root.resolve()
    registration = preflight(root)
    out = root / OUT
    openings = read_rows(out / "openings.jsonl")
    ledger_path = out / "outcomes.jsonl"
    pending_ledger = out / "outcomes.jsonl.pending"
    if pending_ledger.exists():
        pending_rows = read_rows(pending_ledger)
        _verify_committed(root, openings, pending_rows)
        os.replace(pending_ledger, ledger_path)
    ledger = read_rows(ledger_path) if ledger_path.exists() else []
    recovered, pending = resume_plan(root, openings, ledger)
    for index in recovered:
        ledger.extend(_pair_rows(root, index, openings[index]))
        atomic_ledger(ledger_path, ledger)
    for index in pending:
        argv = command(root, index)
        command_record = {
            "opening_index": index,
            "argv": argv,
            "seed": BASE_SEED + index,
            "registration_sha256": sha(out / "registration.json"),
            "suite_sha256": sha(out / "openings.jsonl"),
            "component_assignments": registration["assignments"],
            "component_files": {
                key: value["files"] for key, value in registration["components"].items()
            },
        }
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(command_record, sort_keys=True) + "\n")
        subprocess.run(argv, cwd=root, check=True)
        report_path = out / "batches" / f"report-{index:04d}.json"
        report = json.loads(report_path.read_text())
        report["policy_ablation_composition"] = {
            **registration["assignments"],
            "components": registration["components"],
            "runtime_asset_sha256": registration["runtime_asset_sha256"],
        }
        atomic_json(report_path, report)
        pair = _pair_rows(root, index, openings[index])
        ledger.extend(pair)
        atomic_ledger(ledger_path, ledger)
    if len(ledger) != GAMES:
        raise ValueError("incomplete_arena_ledger")
    result = analyze(ledger)
    atomic_json(out / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.root), indent=2, sort_keys=True))
    else:
        print(
            json.dumps(run(args.root, execute=args.execute), indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
