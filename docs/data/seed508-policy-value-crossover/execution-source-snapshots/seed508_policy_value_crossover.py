"""Exploratory seed508 policy/value output crossover on seed488 openings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed488_strength_evaluation as base

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed508-policy-value-crossover")
SUITE = Path("docs/data/seed488-frozen-strength-evaluation/openings.jsonl")
SEED505 = Path("docs/data/seed505-strength-screen-v3")
E4_SHA = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
ARMS = {
    "P": {"policy": "E4", "value": "initializer"},
    "V": {"policy": "initializer", "value": "E4"},
}
DEPENDENCIES = [
    "ml/alphazero_lite/seed508_policy_value_crossover.py",
    "ml/alphazero_lite/verify_seed508_policy_value_crossover.py",
    "ml/alphazero_lite/test_seed508_policy_value_crossover.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/seed488_strength_evaluation.py",
    "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def atomic_text(path: Path, value: str) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value)
    pending.replace(path)


def command(root: Path, arm: str, index: int, opening: dict[str, Any]) -> list[str]:
    out = root / OUT
    batch = out / "batches"
    batch.mkdir(parents=True, exist_ok=True)
    opener = batch / f"{arm}-opening-{index:04d}.jsonl"
    payload = base.canonical_json(opening) + b"\n"
    if opener.exists() and opener.read_bytes() != payload:
        raise ValueError("opening_batch_conflict")
    opener.write_bytes(payload)
    registration = json.loads(
        (
            root / "docs/data/seed488-frozen-strength-evaluation/registration.json"
        ).read_text()
    )
    ident = base.batch_opening_identity(registration, opening, opener)
    artifact_root = out / "runtime-artifacts"
    side = ARMS[arm]
    argv = base.build_arena_command(
        registration,
        artifact_root,
        opener,
        batch / f"{arm}-report-{index:04d}.json",
        batch / f"{arm}-games-{index:04d}.jsonl",
        ident,
    )
    replacements = {
        "--challenger": str(artifact_root / side["policy"]),
        "--current": str(artifact_root / "initializer"),
        "--challenger-policy-artifact": str(artifact_root / side["policy"]),
        "--challenger-value-artifact": str(artifact_root / side["value"]),
        "--challenger-simulations": "128",
        "--current-simulations": "128",
        "--games": "2",
        "--games-per-opening": "2",
    }
    for flag, val in replacements.items():
        if flag not in argv:
            argv.extend([flag, val])
        else:
            argv[argv.index(flag) + 1] = val
    return argv


def validate_pair(
    root: Path,
    arm: str,
    index: int,
    opening: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
) -> None:
    from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

    out = root / OUT
    batch = out / "batches"
    gp = batch / f"{arm}-games-{index:04d}.jsonl"
    rp = batch / f"{arm}-report-{index:04d}.json"
    op = batch / f"{arm}-opening-{index:04d}.jsonl"
    if not (gp.is_file() and rp.is_file() and op.is_file()):
        raise ValueError(f"complete_pair_shape:{arm}:{index}")
    games, report = rows(gp), json.loads(rp.read_text())
    notes = report.get("notes", {})
    assignment = report.get("seed508_composition", {})
    if (
        op.read_bytes() != base.canonical_json(opening) + b"\n"
        or len(games) != 2
        or report.get("games_played") != 2
    ):
        raise ValueError(f"complete_pair_shape:{arm}:{index}")
    if (
        {int(g["challenger_player"]) for g in games} != {0, 1}
        or notes.get("challenger_simulations") != 128
        or notes.get("current_simulations") != 128
        or notes.get("workers_used") != 1
        or notes.get("base_seed") != 488 + index
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("exact_root_solve_threshold") != 16
        or assignment.get("policy_source") != ARMS[arm]["policy"]
        or assignment.get("value_source") != ARMS[arm]["value"]
        or assignment.get("trunk_policy") != "policy source's own trunk"
        or assignment.get("trunk_value") != "value source's own trunk"
    ):
        raise ValueError(f"complete_pair_runtime:{arm}:{index}")
    arm_rows = [
        r for r in ledger_rows if r["arm"] == arm and int(r["opening_index"]) == index
    ]
    if len(arm_rows) != 2 or {int(r["challenger_player"]) for r in arm_rows} != {0, 1}:
        raise ValueError(f"complete_pair_ledger:{arm}:{index}")
    for row in arm_rows:
        pos = int(row["batch_game_position"])
        game = games[pos]
        if (
            row["batch_game_sha256"] != sha(gp)
            or row["batch_report_sha256"] != sha(rp)
            or row["batch_opening_sha256"] != sha(op)
            or int(game["challenger_player"]) != int(row["challenger_player"])
        ):
            raise ValueError(f"complete_pair_receipt:{arm}:{index}:{pos}")
        winner, _, _ = _replay_game(opening, game, int(row["challenger_player"]))
        if winner != row["winner"]:
            raise ValueError(f"complete_pair_replay:{arm}:{index}:{pos}")


def resume_plan(
    root: Path, openings: list[dict[str, Any]], ledger_rows: list[dict[str, Any]]
) -> list[tuple[str, int]]:
    """Validate every committed pair and return only uncommitted arm/opening pairs."""
    keys = [
        (str(row["arm"]), int(row["opening_index"]), int(row["challenger_player"]))
        for row in ledger_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_committed_game")
    committed = {(arm, index) for arm, index, _ in keys}
    for arm, index in committed:
        if arm not in ARMS or index not in range(len(openings)):
            raise ValueError("committed_pair_assignment")
        validate_pair(root, arm, index, openings[index], ledger_rows)
    pending = []
    batch = root / OUT / "batches"
    for arm in ARMS:
        for index in range(len(openings)):
            if (arm, index) in committed:
                continue
            for prefix, suffix in (("games", "jsonl"), ("report", "json")):
                if (batch / f"{arm}-{prefix}-{index:04d}.{suffix}").exists():
                    raise ValueError(
                        f"interrupted_outcome_batch_preserved:{arm}:{index}"
                    )
            pending.append((arm, index))
    return pending


def run(root: Path = ROOT, *, execute: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if not execute:
        raise ValueError("execution_requires_explicit_--execute")
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if sha(out / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("registration_freeze_mismatch")
    for rel, digest in reg["sources_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_source_changed:{rel}")
    for rel, digest in reg["runtime"]["runtime_identities_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"runtime_identity_changed:{rel}")
    for component in reg["components"].values():
        checkpoint = Path(component["checkpoint"])
        checkpoint_path = (
            out / checkpoint
            if checkpoint.name == "initializer.npz"
            else root / checkpoint
        )
        if sha(checkpoint_path) != component["checkpoint_sha256"]:
            raise ValueError("checkpoint_identity_changed")
        for filename, digest in component["files"].items():
            if sha(out / component["artifact"] / filename) != digest:
                raise ValueError(f"component_artifact_changed:{filename}")
    openings = rows(out / "openings.jsonl")
    ledger = out / "outcomes.jsonl"
    if (out / "outcomes.jsonl.pending").exists():
        raise ValueError("pending_outcome_commit_requires_recovery")
    data = rows(ledger) if ledger.exists() else []
    pending_pairs = resume_plan(root, openings, data)
    for arm, i in pending_pairs:
        opening = openings[i]
        batch = out / "batches"
        argv = command(root, arm, i, opening)
        record = {
            "arm": arm,
            "opening_index": i,
            "argv": argv,
            "policy_source": ARMS[arm]["policy"],
            "value_source": ARMS[arm]["value"],
            "component_artifacts": {
                k: str(out / "runtime-artifacts" / v) for k, v in ARMS[arm].items()
            },
        }
        record["component_checkpoints"] = {
            "E4": {
                "path": "docs/data/seed504/seed504_epoch4.npz",
                "sha256": E4_SHA,
            },
            "initializer": {"path": "initializer.npz", "sha256": INIT_SHA},
        }
        record["component_file_sha256"] = reg["components"]
        record["resolved_assignments"] = {
            "policy": ARMS[arm]["policy"],
            "value": ARMS[arm]["value"],
            "trunk": "each assigned source retains its own evaluator trunk",
        }
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            subprocess.run(argv, cwd=root, check=True)
            gp = batch / f"{arm}-games-{i:04d}.jsonl"
            rp = batch / f"{arm}-report-{i:04d}.json"
            op = batch / f"{arm}-opening-{i:04d}.jsonl"
            report_document = json.loads(rp.read_text())
            report_document["seed508_composition"] = {
                "policy_source": ARMS[arm]["policy"],
                "value_source": ARMS[arm]["value"],
                "trunk_policy": "policy source's own trunk",
                "trunk_value": "value source's own trunk",
                "component_artifacts": record["component_artifacts"],
                "component_checkpoints": record["component_checkpoints"],
                "component_file_sha256": record["component_file_sha256"],
            }
            atomic_json(rp, report_document)
        games = rows(gp)
        if len(games) != 2 or {int(g["challenger_player"]) for g in games} != {
            0,
            1,
        }:
            raise ValueError(f"incomplete_pair_preserved:{arm}:{i}")
        new = []
        for pos, game in enumerate(games):
            winner, _, _ = __import__(
                "ml.alphazero_lite.verify_seed488_strength_evaluation",
                fromlist=["_replay_game"],
            )._replay_game(opening, game, int(game["challenger_player"]))
            new.append(
                {
                    "arm": arm,
                    "opening_index": i,
                    "challenger_player": int(game["challenger_player"]),
                    "winner": winner,
                    "batch_game_position": pos,
                    "batch_game_sha256": sha(gp),
                    "batch_report_sha256": sha(rp),
                    "batch_opening_sha256": sha(op),
                }
            )
        data += new
        atomic_text(
            ledger, "\n".join(json.dumps(r, sort_keys=True) for r in data) + "\n"
        )
    if len(data) != 2048:
        raise ValueError("final_game_count")
    results = analyze(data)
    atomic_json(out / "results.json", results)
    return results


def analyze(data: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(508)

    def points(w: str) -> float:
        return 1.0 if w == "challenger" else 0.5 if w == "draw" else 0.0

    scores = {}
    for arm in ARMS:
        grouped = {}
        for r in data:
            if r["arm"] == arm:
                grouped.setdefault(int(r["opening_index"]), []).append(r)
        per = np.array(
            [sum(points(r["winner"]) for r in grouped[i]) / 2 for i in range(512)]
        )
        descriptive_rng = np.random.default_rng(508 + (1 if arm == "P" else 2))
        sample = per[descriptive_rng.integers(0, 512, (20_000, 512))].mean(axis=1)
        wins = sum(r["arm"] == arm and r["winner"] == "challenger" for r in data)
        draws = sum(r["arm"] == arm and r["winner"] == "draw" for r in data)
        losses = 1024 - wins - draws
        scores[arm] = {
            "games": 1024,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": (wins + 0.5 * draws) / 1024,
            "descriptive_two_sided_95": [
                float(np.quantile(sample, 0.025)),
                float(np.quantile(sample, 0.975)),
            ],
        }
    p = {
        i: [r for r in data if r["arm"] == "P" and int(r["opening_index"]) == i]
        for i in range(512)
    }
    v = {
        i: [r for r in data if r["arm"] == "V" and int(r["opening_index"]) == i]
        for i in range(512)
    }
    diff = np.array(
        [
            sum(points(x["winner"]) for x in p[i]) / 2
            - sum(points(x["winner"]) for x in v[i]) / 2
            for i in range(512)
        ]
    )
    boot = diff[rng.integers(0, 512, (20_000, 512))].mean(axis=1)
    return {
        "standalone": scores,
        "primary": {
            "estimand": "per-opening P score minus V score, each score averages two seat games",
            "mean_difference": float(diff.mean()),
            "bootstrap_resamples": 20_000,
            "seed": 508,
            "two_sided_95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "interpretation": "P supported"
            if np.quantile(boot, 0.025) > 0
            else "V supported"
            if np.quantile(boot, 0.975) < 0
            else "inconclusive",
        },
        "historical_seed505_e4_vs_initializer": {
            "score": 0.50830078125,
            "wins": 474,
            "draws": 93,
            "losses": 457,
            "descriptive_two_sided_95": [0.48583984375, 0.5302734375],
            "one_sided_lower_95": 0.48974609375,
            "classification": "historical descriptive result; not additive decomposition or promotion gate",
        },
        "promotion": False,
        "fresh_confirmation": False,
    }


def prepare(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(
        (out / f).exists()
        for f in ("outcomes.jsonl", "outcomes.jsonl.pending", "commands.jsonl")
    ):
        raise ValueError("seed508_execution_already_started")
    prior_freeze = None
    if (out / "execution-freeze.json").is_file():
        prior_freeze = {
            "registration_sha256": sha(out / "registration.json"),
            "freeze_sha256": sha(out / "execution-freeze.json"),
        }
        archive = out / "superseded-pre-execution-freeze-001"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "registration.json", archive / "registration.json")
        shutil.copyfile(
            out / "execution-freeze.json", archive / "execution-freeze.json"
        )
    historic = root / SEED505
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
    if (
        sha(root / "docs/data/seed504/seed504_epoch4.npz") != E4_SHA
        or sha(historic / "initializer.npz") != INIT_SHA
    ):
        raise ValueError("frozen_checkpoint_identity")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / SUITE, out / "openings.jsonl")
    shutil.copyfile(historic / "initializer.npz", out / "initializer.npz")
    for name in ("E4", "initializer"):
        destination = out / "runtime-artifacts" / name
        if not destination.exists():
            shutil.copytree(historic / "runtime-artifacts" / name, destination)
    suite_reg = json.loads(
        (
            root / "docs/data/seed488-frozen-strength-evaluation/registration.json"
        ).read_text()
    )
    sources = {f: sha(root / f) for f in DEPENDENCIES}
    artifacts = {
        name: {
            f: sha(out / "runtime-artifacts" / name / f)
            for f in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        for name in ("E4", "initializer")
    }
    registration = {
        "schema": "seed508-policy-value-crossover-v1",
        "status": "frozen_before_execution",
        "pre_execution_supersession": prior_freeze,
        "interpretation": "exploratory diagnostic; not additive decomposition, promotion, fresh confirmation, or advancement gate",
        "suite": {
            "source": str(SUITE),
            "sha256": sha(out / "openings.jsonl"),
            "openings": 512,
            "games_per_arm": 1024,
            "reused_from_seed": 488,
            "historical_reuse_disclosed": True,
            "historical_outcomes_used": False,
        },
        "components": {
            "E4": {
                "checkpoint": "docs/data/seed504/seed504_epoch4.npz",
                "checkpoint_sha256": E4_SHA,
                "artifact": "runtime-artifacts/E4",
                "files": artifacts["E4"],
            },
            "initializer": {
                "checkpoint": "initializer.npz",
                "checkpoint_sha256": INIT_SHA,
                "artifact": "runtime-artifacts/initializer",
                "files": artifacts["initializer"],
            },
        },
        "arms": {
            "P": {"policy_source": "E4", "value_source": "initializer"},
            "V": {"policy_source": "initializer", "value_source": "E4"},
        },
        "runtime": {
            "simulations_per_side": 128,
            "c_puct": 1.25,
            "workers": 1,
            "seed": "488 + parent opening index",
            "seed_contract": "azlite_eval_seed_v2",
            "max_moves": 200,
            "search_options": suite_reg["runtime"]["search_options"],
            "exact_root": suite_reg["runtime"]["exact_root"],
            "runtime_identities_sha256": suite_reg["runtime"]["identities_sha256"],
        },
        "analysis": {
            "primary": "opening-level P score minus V score; each arm opening score averages both seats",
            "bootstrap_resamples": 20000,
            "seed": 508,
            "interval": "two-sided 95th percentile",
            "standalone_intervals": "descriptive paired-opening percentile intervals",
        },
        "limitations": [
            "seed488's 512-opening suite was reused and previously examined",
            "seed461 continuation-population overlap remains unresolved",
            "seed505 full-E4 screen is historical descriptive evidence only",
            "not an additive decomposition or promotion gate",
        ],
        "sources_sha256": sources,
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": __import__("numpy").__version__,
        },
    }
    atomic_json(out / "registration.json", registration)
    freeze = {
        "status": "frozen_before_execution",
        "registration_sha256": sha(out / "registration.json"),
        "suite_sha256": sha(out / "openings.jsonl"),
        "source_sha256": sources,
        "component_artifacts": artifacts,
    }
    atomic_json(out / "execution-freeze.json", freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(run(execute=True), indent=2, sort_keys=True))
    elif args.prepare:
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    else:
        parser.error("choose --prepare or --execute")


if __name__ == "__main__":
    main()
