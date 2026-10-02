"""Register and evaluate the frozen seed397 O0 E1/E4 comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import checkpoint_trajectory_diagnostic as trajectory
from ml.alphazero_lite import run_seed461_order_exploratory as exploratory
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.opening_exclusion_contract import opening_identities
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed397-o0-e1-e4-registration.json"
SUITE = DATA / "seed397-o0-e1-e4-openings.jsonl"
EXCLUSIONS = DATA / "seed397-o0-e1-e4-exclusions.json"
BIND = DATA / "seed397-o0-e1-e4-evaluation-binding.json"
RESULT = DATA / "seed397-o0-e1-e4-results.json"
ACCOUNTING = DATA / "seed397-o0-e1-e4-game-outcome-accounting.jsonl"
MATRIX = DATA / "seed397-o0-e1-e4-paired-opening-matrix.json"
RECOVERY = DATA / "seed461-o0-checkpoint-recovery-provenance.json"
WORK = ROOT / ".tmp/seed397-o0-e1-e4"
PUBLISHED_EXCLUSIONS = (
    DATA / "order38615-a5-frozen-diagnostic-v4/opening-exclusion-manifest.json"
)
A5_SUITE_PATHS = [
    DATA / "order38615-a5-frozen-diagnostic-v4/seed395-openings-v2.jsonl",
    DATA / "order38615-a5-frozen-diagnostic-v4/seed396-openings-v2.jsonl",
]
EXPECTED = {
    "E1": "643407f0b070acc1603e90e14e4ad8fd4579e4e34e3c109d9b0691f08267a1dc",
    "E2": "73914b801e61be7eada0366619fbc2defb23999fb163ef0da49134594ec44488",
    "E3": "9d3db177d22f12b0c8952233f3fc6c5dad5b0529f3990932699e8839afa0d2b8",
    "E4": "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4",
}
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replay_keys(paths: list[Path]) -> set[str]:
    identities: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                identities.add(
                    suites.canonical_key(trajectory.state_from_row(json.loads(line)))
                )
    return identities


def register() -> None:
    if any(path.exists() for path in (REG, SUITE, EXCLUSIONS, BIND, RESULT)):
        raise RuntimeError("seed397_registration_or_results_already_exist")
    published = json.loads(PUBLISHED_EXCLUSIONS.read_text(encoding="utf-8"))
    a5_keys = set(published["actual_state_identities"])
    if len(a5_keys) != int(published["actual_state_count"]):
        raise RuntimeError("a5_exclusion_manifest_identity_count_mismatch")
    replay_paths, replay_records, parent = exploratory.inputs()
    for path, record in zip(replay_paths, replay_records, strict=True):
        if not path.is_file() or sha(path) != record["sha256"]:
            raise RuntimeError(f"original_training_replay_hash_mismatch:{path}")
    if not parent.is_file():
        raise RuntimeError(f"original_parent_initializer_missing:{parent}")
    if (
        sha(parent)
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise RuntimeError("original_parent_initializer_hash_mismatch")
    train_keys = replay_keys(replay_paths)
    for path in A5_SUITE_PATHS:
        if not path.is_file():
            raise RuntimeError(f"prior_a5_suite_missing:{path}")
    prior_suites = set()
    for path in A5_SUITE_PATHS:
        rows = suites.load_suite_jsonl(str(path))
        _, identities = opening_identities(rows)
        prior_suites.update(identities)
    excluded = a5_keys | train_keys | prior_suites
    prefixes = suites.enumerate_legal_prefixes(8)
    population, _, _ = suites.deduplicate_openings(prefixes)
    eligible = [
        row
        for row in suites.stratify_openings(population)
        if suites.canonical_key(row["state"]) not in excluded
        and int(row["pit_sum"]) > 32
        and not KalahGame.from_state(row["state"]).over()
        and len(row.get("prefix_moves", [])) <= 8
    ]
    selected = suites.select_diverse(eligible, 512, 397)
    selected_ids = [suites.canonical_key(row["state"]) for row in selected]
    if (
        len(selected) != 512
        or len(set(selected_ids)) != 512
        or set(selected_ids) & excluded
    ):
        raise RuntimeError("seed397_opening_selection_contract_failure")
    selected = [suites.export_arena_entry(row) for row in selected]
    suites.write_suite_jsonl(selected, str(SUITE))
    suite_rows = suites.load_suite_jsonl(str(SUITE))
    suites.validate_arena_entries(suite_rows)
    write(
        EXCLUSIONS,
        {
            "schema": "seed397-opening-exclusions-v1",
            "sources": [
                {
                    "path": str(PUBLISHED_EXCLUSIONS.relative_to(ROOT)),
                    "sha256": sha(PUBLISHED_EXCLUSIONS),
                    "kind": "verified_a5_complete_exclusion_union",
                    "states": len(a5_keys),
                },
                *[
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha(path),
                        "kind": "seed395_or_396_frozen_suite",
                        "states": len(suites.load_suite_jsonl(str(path))),
                    }
                    for path in A5_SUITE_PATHS
                ],
                *[
                    {
                        "path": record["path"],
                        "sha256": record["sha256"],
                        "kind": "original_seed461_training_replay",
                        "name": record["name"],
                    }
                    for record in replay_records
                ],
            ],
            "a5_union_state_count": len(a5_keys),
            "training_replay_union_state_count": len(train_keys),
            "seed395_396_start_count": len(prior_suites),
            "excluded_union_state_count": len(excluded),
            "selected_exclusion_overlap": 0,
        },
    )
    candidates = []
    binding = json.loads(
        (
            DATA / "seed461-batch-order-sensitivity-candidate-artifact-binding.json"
        ).read_text()
    )
    for row in binding["candidates"]:
        if row["trajectory"] != "O0" or row["epoch"] not in {"E1", "E4"}:
            continue
        checkpoint, artifact = Path(row["checkpoint"]), Path(row["artifact"])
        if (
            sha(checkpoint) != EXPECTED[row["epoch"]]
            or sha(artifact / "model.npz") != EXPECTED[row["epoch"]]
        ):
            raise RuntimeError(f"bound_checkpoint_identity_mismatch:{row['epoch']}")
        actual = {
            "checkpoint_sha256": sha(checkpoint),
            "model_sha256": sha(artifact / "model.npz"),
            "weights_sha256": sha(artifact / "weights.json"),
            "metadata_sha256": sha(artifact / "metadata.json"),
            "search_policy_sha256": sha(artifact / "search_policy.json"),
        }
        for field in actual:
            expected = (
                row["weights_sha256"]
                if field == "weights_sha256"
                else row["metadata_sha256"]
                if field == "metadata_sha256"
                else row["runtime_contract"].get("runtime_search_policy_sha256")
                if field == "search_policy_sha256"
                else row["checkpoint_sha256"]
            )
            if actual[field] != expected:
                raise RuntimeError(f"candidate_{field}_mismatch:{row['epoch']}")
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT, challenger_artifact=artifact
        )
        if contract != row["runtime_contract"]:
            raise RuntimeError(f"runtime_contract_mismatch:{row['epoch']}")
        candidates.append(
            {
                "epoch": row["epoch"],
                "artifact": str(artifact),
                **actual,
                "runtime_contract": contract,
            }
        )
    if {row["epoch"] for row in candidates} != {"E1", "E4"}:
        raise RuntimeError("bound_o0_candidates_incomplete")
    write(
        REG,
        {
            "schema": "seed397-o0-e1-e4-registration-v1",
            "status": "registered_before_games",
            "selection_seed": 397,
            "suite": {
                "path": str(SUITE.relative_to(ROOT)),
                "sha256": sha(SUITE),
                "openings": 512,
                "plies_at_most": 8,
                "nonterminal": True,
                "pit_sum_above": 32,
            },
            "exclusions": {
                "path": str(EXCLUSIONS.relative_to(ROOT)),
                "sha256": sha(EXCLUSIONS),
            },
            "candidates": candidates,
            "opponent": {
                "artifact": str(OPPONENT),
                "checkpoint_sha256": "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1",
                "weights_sha256": sha(OPPONENT / "weights.json"),
                "metadata_sha256": sha(OPPONENT / "metadata.json"),
                "search_policy_sha256": sha(OPPONENT / "search_policy.json"),
            },
            "evaluation": {
                "games_per_checkpoint": 1024,
                "games_total": 2048,
                "games_per_opening": 2,
                "both_seats": True,
                "simulations_per_side": 384,
                "c_puct": 1.25,
                "arena_base_seed": 397,
                "seed_contract": "azlite_eval_seed_v2",
                "native_runtime_unchanged": True,
                "extensions": "forbidden",
            },
            "analysis": {
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 397,
                "cluster": "opening",
                "pairing": "same opening and both seats",
                "interval": "95% percentile",
            },
            "decision": {
                "useful_early_gain": "E1 score >= 0.55 and 95% lower bound > 0.50",
                "subsequent_weakening": "paired mean E1-E4 >= 0.03 and 95% lower bound > 0",
            },
        },
    )
    write(
        BIND,
        {
            "schema": "seed397-o0-e1-e4-evaluation-binding-v1",
            "registration_sha256": sha(REG),
            "suite_sha256": sha(SUITE),
            "exclusions_sha256": sha(EXCLUSIONS),
            "status": "registered_before_games",
            "reports": {},
        },
    )


def run() -> None:
    if not REG.is_file() or not BIND.is_file():
        raise RuntimeError("seed397_registration_required_before_games")
    reg, binding = json.loads(REG.read_text()), json.loads(BIND.read_text())
    if (
        binding["registration_sha256"] != sha(REG)
        or binding["suite_sha256"] != sha(SUITE)
        or sha(SUITE) != reg["suite"]["sha256"]
    ):
        raise RuntimeError("seed397_preregistration_binding_mismatch")
    for cand in reg["candidates"]:
        tag = cand["epoch"]
        report, games = WORK / f"{tag}.json", WORK / f"{tag}-games.jsonl"
        if report.exists() or games.exists():
            raise RuntimeError(f"seed397_output_already_exists:{tag}")
        binding["reports"][tag] = {
            "state": "running",
            "report": str(report),
            "games": str(games),
        }
        write(BIND, binding)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                cand["artifact"],
                "--current",
                str(OPPONENT),
                "--games",
                "1024",
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(SUITE),
                "--suite-sha256",
                sha(SUITE),
                "--challenger-simulations",
                "384",
                "--current-simulations",
                "384",
                "--seed",
                "397",
                "--workers",
                "24",
                "--c-puct",
                "1.25",
                "--seed-contract",
                "azlite_eval_seed_v2",
                "--game-jsonl",
                str(games),
                "--out",
                str(report),
            ],
            cwd=ROOT,
            check=True,
        )
        binding["reports"][tag] = {
            "state": "completed",
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
        }
        write(BIND, binding)
    binding["status"] = "completed_2048_games"
    write(BIND, binding)
    analyze()


def analyze() -> None:
    rows_by_epoch = {}
    for epoch in ("E1", "E4"):
        rows_by_epoch[epoch] = [
            json.loads(line)
            for line in (WORK / f"{epoch}-games.jsonl").read_text().splitlines()
            if line
        ]
        if len(rows_by_epoch[epoch]) != 1024:
            raise RuntimeError(f"outcome_count_mismatch:{epoch}")
    suite_rows = suites.load_suite_jsonl(str(SUITE))
    _, suite_identities = opening_identities(suite_rows)
    for epoch in ("E1", "E4"):
        for i, row in enumerate(rows_by_epoch[epoch]):
            row.update(
                {
                    "checkpoint": epoch,
                    "opening_index": int(row["opening_index"]),
                    "game_index": i,
                }
            )
    indexed = {
        epoch: {row["opening_index"]: [] for row in rows_by_epoch[epoch]}
        for epoch in rows_by_epoch
    }
    for epoch in rows_by_epoch:
        for row in rows_by_epoch[epoch]:
            indexed[epoch][row["opening_index"]].append(
                1.0
                if row["winner"] == "challenger"
                else 0.5
                if row["winner"] == "draw"
                else 0.0
            )
        seat_pairs = {
            opening: {
                int(row["challenger_player"])
                for row in rows_by_epoch[epoch]
                if int(row["opening_index"]) == opening
            }
            for opening in indexed[epoch]
        }
        if any(
            len(indexed_games) != 2 for indexed_games in indexed[epoch].values()
        ) or any(seats != {0, 1} for seats in seat_pairs.values()):
            raise RuntimeError(f"seat_pairing_mismatch:{epoch}")
    keys = sorted(indexed["E1"])
    if keys != sorted(indexed["E4"]):
        raise RuntimeError("checkpoint_opening_pair_mismatch")
    scores = {
        epoch: np.asarray([np.mean(indexed[epoch][key]) for key in keys])
        for epoch in ("E1", "E4")
    }
    delta = scores["E1"] - scores["E4"]
    rng = np.random.default_rng(397)
    indexes = rng.integers(0, len(keys), size=(10_000, len(keys)))
    e1_draws = scores["E1"][indexes].mean(axis=1)
    diff_draws = delta[indexes].mean(axis=1)
    outcomes = {}
    accounting_rows = []
    for epoch in ("E1", "E4"):
        rows = rows_by_epoch[epoch]
        for row in rows:
            accounting_rows.append(
                {
                    "checkpoint": epoch,
                    "game_index": int(row["game_index"]),
                    "opening_index": int(row["opening_index"]),
                    "opening_state_hash": suite_identities[int(row["opening_index"])],
                    "challenger_player": int(row["challenger_player"]),
                    "winner": str(row["winner"]),
                    "margin": int(row["margin"]),
                    "game_length": int(row["game_length"]),
                }
            )
        counts = {
            result: sum(row["winner"] == result for row in rows)
            for result in ("challenger", "draw", "current")
        }
        outcomes[epoch] = {
            "games": len(rows),
            "wins": counts["challenger"],
            "draws": counts["draw"],
            "losses": counts["current"],
            "score": float(scores[epoch].mean()),
            "by_seat": {
                str(seat): {
                    "games": sum(int(row["challenger_player"]) == seat for row in rows),
                    "score": float(
                        np.mean(
                            [
                                1.0
                                if row["winner"] == "challenger"
                                else 0.5
                                if row["winner"] == "draw"
                                else 0.0
                                for row in rows
                                if int(row["challenger_player"]) == seat
                            ]
                        )
                    ),
                }
                for seat in (0, 1)
            },
        }
    ACCOUNTING.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accounting_rows),
        encoding="utf-8",
    )
    matrix = {
        "schema": "seed397-o0-e1-e4-paired-opening-matrix-v1",
        "registration_sha256": sha(REG),
        "accounting_sha256": sha(ACCOUNTING),
        "opening_state_hashes": [suite_identities[index] for index in keys],
        "opening_indices": keys,
        "E1": scores["E1"].tolist(),
        "E4": scores["E4"].tolist(),
        "E1_minus_E4": delta.tolist(),
    }
    write(MATRIX, matrix)
    result = {
        "schema": "seed397-o0-e1-e4-results-v1",
        "registration_sha256": sha(REG),
        "binding_sha256": sha(BIND),
        "recovery_provenance_sha256": sha(RECOVERY),
        "accounting_sha256": sha(ACCOUNTING),
        "paired_matrix_sha256": sha(MATRIX),
        "outcomes": outcomes,
        "paired_opening_matrix": matrix,
        "analysis": {
            "bootstrap_resamples": 10000,
            "seed": 397,
            "E1_score_interval_95": [
                float(x) for x in np.quantile(e1_draws, [0.025, 0.975])
            ],
            "E1_minus_E4_mean": float(delta.mean()),
            "E1_minus_E4_interval_95": [
                float(x) for x in np.quantile(diff_draws, [0.025, 0.975])
            ],
        },
    }
    ci = result["analysis"]["E1_score_interval_95"]
    di = result["analysis"]["E1_minus_E4_interval_95"]
    result["classification"] = {
        "useful_early_gain": bool(scores["E1"].mean() >= 0.55 and ci[0] > 0.50),
        "subsequent_weakening": bool(delta.mean() >= 0.03 and di[0] > 0.0),
    }
    write(RESULT, result)


if __name__ == "__main__":
    if "register" in sys.argv:
        register()
    elif "run" in sys.argv:
        run()
    elif "analyze" in sys.argv:
        analyze()
    else:
        raise SystemExit("usage: run_seed397_o0_e1_e4.py {register|run|analyze}")
