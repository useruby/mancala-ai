"""Frozen exploratory seed505 arena comparing seed504 E4 with its initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite import run_seed504
from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed505-strength-screen-v3")
HISTORICAL = Path("docs/data/seed488-frozen-strength-evaluation")
SEED504 = Path("docs/data/seed504")
SIMULATIONS = 128
SEED = 488
ANALYSIS_SEED = 505
GAME_COUNT = 1024
E4_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(root: Path = ROOT) -> dict[str, Any]:
    """Freeze registered inputs, command dependencies, suite, artifacts, and analysis."""
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(
        (out / name).exists()
        for name in (
            "registration.json",
            "execution-freeze.json",
            "outcomes.jsonl",
            "outcomes.jsonl.pending",
        )
    ):
        raise ValueError("seed505_output_already_frozen_or_executed")
    if sha(root / SEED504 / "seed504_epoch4.npz") != E4_SHA256:
        raise ValueError("seed504_e4_hash")
    if sha(root / ".tmp/seed500-current-checkpoint.npz") != INITIALIZER_SHA256:
        raise ValueError("seed504_initializer_hash")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        root / ".tmp/seed500-current-checkpoint.npz", out / "initializer.npz"
    )
    historical = root / HISTORICAL
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(root),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if json.loads(verified.stdout).get("status") != "verified_complete":
        raise ValueError("seed488_suite_verification")
    shutil.copyfile(historical / "openings.jsonl", out / "openings.jsonl")
    artifact_dir = out / "runtime-artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for arm, checkpoint in (
        ("E4", root / SEED504 / "seed504_epoch4.npz"),
        ("initializer", out / "initializer.npz"),
    ):
        if not (artifact_dir / arm / "model.npz").is_file():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml.alphazero_lite.export_artifact",
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(artifact_dir / arm),
                    "--version",
                    f"seed505-{arm}",
                    "--model-type",
                    "residual_v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=root,
                check=True,
            )
        shutil.copyfile(
            root / "model-artifact/current/search_policy.json",
            artifact_dir / arm / "search_policy.json",
        )
        import numpy as np

        with (
            np.load(checkpoint, allow_pickle=False) as expected,
            np.load(artifact_dir / arm / "model.npz", allow_pickle=False) as exported,
        ):
            if set(expected.files) != set(exported.files) or any(
                not np.array_equal(expected[name], exported[name])
                for name in expected.files
            ):
                raise ValueError(f"exported_checkpoint_mismatch:{arm}")
    artifacts = {
        arm: {
            name: sha(artifact_dir / arm / name)
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        for arm in ("E4", "initializer")
    }
    (out / "runtime-artifacts.json").write_text(
        json.dumps(artifacts, indent=2, sort_keys=True) + "\n"
    )
    source_files = [
        "ml/alphazero_lite/seed505_strength_screen.py",
        "ml/alphazero_lite/verify_seed505_strength_screen.py",
        "ml/alphazero_lite/test_seed505_strength_screen.py",
        "ml/alphazero_lite/arena.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/classic_mcts.py",
        "ml/alphazero_lite/native_exact_root_tablebase.py",
        "ml/alphazero_lite/exact_root_decision.py",
        "ml/alphazero_lite/runtime_search_policy.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/search_ablation.py",
        "ml/alphazero_lite/root_prior_transforms.py",
        "ml/alphazero_lite/policy_prior_localization.py",
        "ml/alphazero_lite/endgame_tablebase.py",
        "ml/alphazero_lite/value_transforms.py",
        "ml/alphazero_lite/export_artifact.py",
        "ml/alphazero_lite/opening_cache.py",
        "ml/alphazero_lite/report_validation.py",
        "ml/alphazero_lite/shadow_root_q.py",
        "ml/alphazero_lite/seed488_strength_evaluation.py",
        "ml/alphazero_lite/seed482_fresh_confirmation.py",
        "ml/alphazero_lite/evaluation_seed_contract.py",
        "ml/alphazero_lite/opening_exclusion_contract.py",
        "ml/alphazero_lite/build_opening_suite.py",
    ]
    source_hashes = {name: sha(root / name) for name in source_files}
    training, _, _ = run_seed504.split()
    training_canonical = {run_seed504.keys(row)[0] for row in training}
    training_float32 = {run_seed504.keys(row)[1] for row in training}
    suite = read_rows(out / "openings.jsonl")
    suite_inputs = [
        tuple(map(float, encode_state(row["state"], input_encoding="kalah_v3")))
        for row in suite
    ]
    overlap = {
        "training_canonical_identity_unique": len(training_canonical),
        "training_production_float32_input_unique": len(training_float32),
        "suite_openings": len(suite),
        "canonical_root_overlap_count": sum(
            tuple(value[:15]) in training_canonical for value in suite_inputs
        ),
        "production_float32_root_overlap_count": sum(
            value in training_float32 for value in suite_inputs
        ),
        "denominator": len(suite),
        "interpretation": "opening-root overlap against seed504 training canonical board identities and complete production float32 model inputs; suite reuse means this is not an untouched or historically unseen holdout",
    }
    registration = {
        "schema": "seed505-exploratory-strength-screen-v1",
        "status": "frozen_before_execution",
        "interpretation": "exploratory strength screen on seed488's previously examined suite; not untouched, historically unseen, fresh confirmation, or promotion",
        "suite_reuse": {
            "seed": 488,
            "opening_count": 512,
            "games": 1024,
            "reuse_disclosed": True,
            "historical_outcomes_reused": False,
            "suite_sha256": sha(out / "openings.jsonl"),
        },
        "challenger": {
            "checkpoint": "docs/data/seed504/seed504_epoch4.npz",
            "sha256": E4_SHA256,
        },
        "initializer": {
            "checkpoint": "initializer.npz (archived from seed504's recorded source)",
            "sha256": INITIALIZER_SHA256,
        },
        "runtime": {
            "simulations_per_side": SIMULATIONS,
            "c_puct": 1.25,
            "workers": 1,
            "max_moves": 200,
            "base_seed": "488 + parent opening index",
            "seed_contract": "azlite_eval_seed_v2",
            "search_options": json.loads(
                (historical / "registration.json").read_text()
            )["runtime"]["search_options"],
            "exact_root": json.loads((historical / "registration.json").read_text())[
                "runtime"
            ]["exact_root"],
            "runtime_identities_sha256": json.loads(
                (historical / "registration.json").read_text()
            )["runtime"]["identities_sha256"],
        },
        "runtime_artifacts": artifacts,
        "training_root_overlap": overlap,
        "analysis": {
            "score": "(wins + 0.5 * draws) / 1024",
            "bootstrap": "20,000 paired-opening percentile resamples",
            "seed": ANALYSIS_SEED,
            "pass": "one-sided 95% lower bound > 0.50",
            "two_sided_interval": "descriptive 95% percentile interval",
        },
        "limitations": [
            "suite reused from seed488, whose outcomes were previously examined",
            "opening-root overlap against seed504 training states will be quantified",
            "seed461 continuation overlap limitation retained",
            "passing supports proposing a later 1,200-simulation test only; no promotion or fresh confirmation",
        ],
        "source_sha256": source_hashes,
        "environment": {"python": sys.version, "platform": sys.platform},
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    freeze = {
        "status": "frozen_before_execution",
        "registration_sha256": sha(out / "registration.json"),
        "suite_sha256": sha(out / "openings.jsonl"),
        "artifact_manifest_sha256": sha(out / "runtime-artifacts.json"),
        "sources_sha256": source_hashes,
    }
    (out / "execution-freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    return freeze


def command(root: Path, index: int, opening: dict[str, Any]) -> list[str]:
    out = root / OUT
    batch = out / "batches"
    batch.mkdir(exist_ok=True)
    opener = batch / f"opening-{index:04d}.jsonl"
    payload = base.canonical_json(opening) + b"\n"
    if opener.exists() and opener.read_bytes() != payload:
        raise ValueError("batch_opening_conflict")
    opener.write_bytes(payload)
    historical = json.loads((root / HISTORICAL / "registration.json").read_text())
    identity = base.batch_opening_identity(historical, opening, opener)
    result = base.build_arena_command(
        historical,
        out / "runtime-artifacts",
        opener,
        batch / f"report-{index:04d}.json",
        batch / f"games-{index:04d}.jsonl",
        identity,
    )
    result[result.index("--challenger") + 1] = str(out / "runtime-artifacts/E4")
    result[result.index("--current") + 1] = str(out / "runtime-artifacts/initializer")
    for flag in ("--challenger-simulations", "--current-simulations"):
        result[result.index(flag) + 1] = str(SIMULATIONS)
    return result


def run(root: Path = ROOT, *, execute: bool = False) -> None:
    root = root.resolve()
    out = root / OUT
    if not execute:
        raise ValueError("execution_requires_explicit_--execute")
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if (
        freeze.get("status") != "frozen_before_execution"
        or sha(out / "registration.json") != freeze["registration_sha256"]
    ):
        raise ValueError("execution_freeze_mismatch")
    for path, digest in registration["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"frozen_source_changed:{path}")
    for path, digest in registration["runtime"]["runtime_identities_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"runtime_identity:{path}")
    if (
        sha(root / SEED504 / "seed504_epoch4.npz") != E4_SHA256
        or sha(out / "initializer.npz") != INITIALIZER_SHA256
    ):
        raise ValueError("checkpoint_identity")
    openings = read_rows(out / "openings.jsonl")
    ledger = out / "outcomes.jsonl"
    pending = out / "outcomes.jsonl.pending"
    if pending.exists():
        raise ValueError("pending_outcome_commit_requires_recovery")
    rows = read_rows(ledger) if ledger.exists() else []
    seen = {int(row["opening_index"]) for row in rows}
    if len(seen) != len(
        {(int(row["opening_index"]), int(row["F_player"])) for row in rows}
    ) or any(sum(int(row["opening_index"]) == i for row in rows) != 2 for i in seen):
        raise ValueError("partial_or_duplicate_committed_pair")
    from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

    for i in seen:
        pair = [row for row in rows if int(row["opening_index"]) == i]
        if {int(row["F_player"]) for row in pair} != {0, 1}:
            raise ValueError(f"committed_pair_seats:{i}")
        opening = openings[i]
        batch = out / "batches"
        game_path = batch / f"games-{i:04d}.jsonl"
        report_path = batch / f"report-{i:04d}.json"
        opener = batch / f"opening-{i:04d}.jsonl"
        games = read_rows(game_path)
        report = json.loads(report_path.read_text())
        notes = report.get("notes", {})
        if (
            opener.read_bytes() != base.canonical_json(opening) + b"\n"
            or len(games) != 2
            or notes.get("challenger_simulations") != SIMULATIONS
            or notes.get("current_simulations") != SIMULATIONS
            or notes.get("workers_used") != 1
            or notes.get("base_seed") != SEED + i
            or notes.get("search_profile", {}).get("c_puct") != 1.25
        ):
            raise ValueError(f"committed_runtime_or_opening:{i}")
        for row in pair:
            pos = int(row["batch_game_position"])
            game = games[pos]
            if (
                row.get("batch_game_sha256") != sha(game_path)
                or row.get("batch_report_sha256") != sha(report_path)
                or row.get("batch_opening_sha256") != sha(opener)
                or int(game["challenger_player"]) != int(row["F_player"])
            ):
                raise ValueError(f"committed_binding:{i}")
            _replay_game(opening, game, int(row["F_player"]))
            expected = {"challenger": "E4", "current": "initializer"}.get(
                game["winner"], game["winner"]
            )
            if row["winner"] != expected:
                raise ValueError(f"committed_outcome:{i}")
    for opening in openings:
        i = int(opening["opening_index"])
        batch = out / "batches"
        if i in seen:
            continue
        games_path, report_path = (
            batch / f"games-{i:04d}.jsonl",
            batch / f"report-{i:04d}.json",
        )
        if games_path.exists() or report_path.exists():
            raise ValueError(f"interrupted_outcome_batch_preserved:{i}")
        argv = command(root, i, opening)
        command_record = {
            "opening_index": i,
            "argv": argv,
            "python": sys.version,
            "settings": {
                "challenger": "seed504 E4",
                "current": "initializer",
                "games": 2,
                "simulations_per_side": 128,
                "c_puct": 1.25,
                "workers": 1,
                "seed": SEED + i,
                "seed_contract": "azlite_eval_seed_v2",
                "move_cap": 200,
                "random_opening_plies": 0,
            },
        }
        with (out / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(command_record, sort_keys=True) + "\n")
        subprocess.run(argv, cwd=root, check=True)
        games = read_rows(games_path)
        if len(games) != 2 or {int(g["challenger_player"]) for g in games} != {0, 1}:
            raise ValueError(f"incomplete_pair_preserved:{i}")
        normalized = []
        for position, game in enumerate(games):
            seat = int(game["challenger_player"])
            _replay_game(opening, game, seat)
            normalized.append(
                {
                    **game,
                    "opening_index": i,
                    "opening_state_hash": opening["state_hash"],
                    "F_player": seat,
                    "batch_game_position": position,
                    "batch_game_sha256": sha(games_path),
                    "batch_report_sha256": sha(report_path),
                    "batch_opening_sha256": sha(batch / f"opening-{i:04d}.jsonl"),
                }
            )
        report = json.loads(report_path.read_text())
        notes = report.get("notes", {})
        if (
            notes.get("challenger_simulations") != SIMULATIONS
            or notes.get("current_simulations") != SIMULATIONS
            or notes.get("workers_used") != 1
            or notes.get("base_seed") != SEED + i
            or notes.get("search_profile", {}).get("c_puct") != 1.25
        ):
            raise ValueError(f"batch_runtime_contract:{i}")
        pending.write_text(
            (ledger.read_text() if ledger.exists() else "")
            + "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized)
        )
        pending.replace(ledger)
        rows.extend(normalized)
    if len(rows) != GAME_COUNT:
        raise ValueError("final_game_count")
    winners = [row["winner"] for row in rows]
    wins, draws, losses = (
        winners.count("challenger"),
        winners.count("draw"),
        winners.count("current"),
    )
    opening_scores = [
        sum(
            1.0
            if r["winner"] == "challenger"
            else 0.5
            if r["winner"] == "draw"
            else 0.0
            for r in rows
            if int(r["opening_index"]) == i
        )
        / 2
        for i in range(512)
    ]
    import numpy as np

    rng = np.random.default_rng(ANALYSIS_SEED)
    samples = np.asarray(opening_scores)[rng.integers(0, 512, size=(20_000, 512))].mean(
        axis=1
    )
    analysis = {
        "games": GAME_COUNT,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / GAME_COUNT,
        "bootstrap": {
            "replicates": 20_000,
            "seed": ANALYSIS_SEED,
            "one_sided_lower_95": float(np.quantile(samples, 0.05)),
            "two_sided_95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        },
        "pass": bool(np.quantile(samples, 0.05) > 0.5),
        "promotion": False,
        "fresh_confirmation": False,
    }
    (out / "results.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare() if args.prepare else run(execute=True), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
