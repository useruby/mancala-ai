"""Retrospective paired first-action budget diagnostic from the #556 archive."""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import (
    ARCHIVE,
    COMMIT,
    canonical_hash,
    git_blob,
    load_authenticated_archive,
    sha256,
)
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed560-paired-first-action/run-0004"
P_ARTIFACT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4"
)
INITIALIZER = (
    ROOT
    / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/initializer"
)
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}


def candidates() -> tuple[list[dict], dict]:
    openings, records, receipt = load_authenticated_archive()
    eligible = []
    for source in records:
        game_row = source["game_row"]
        game = KalahGame.from_state(openings[source["opening_index"]]["state"])
        moves = [int(x) for x in game_row["trajectory"].split(",") if x]
        for ply, move in enumerate(moves):
            if game.over():
                raise ValueError("source trajectory continues after terminal")
            legal = [game.pit_index(a) for a in game.possible_moves()]
            if 33 <= sum(game.pits) <= 44 and len(legal) >= 3:
                eligible.append(
                    {
                        "opening_index": source["opening_index"],
                        "game_within_opening": int(game_row["game_within_opening"]),
                        "source_game_index_local": int(game_row["game_index"]),
                        "source_player_to_move": game.current_player,
                        "source_side": "T512"
                        if game.current_player == int(game_row["challenger_player"])
                        else "P",
                        "ply": ply,
                        "state": game.to_state(),
                        "state_hash": canonical_hash(game.to_state()),
                        "active_pit_stones": sum(game.pits),
                        "legal_absolute_actions": legal,
                        "opening_state_hash": openings[source["opening_index"]][
                            "state_hash"
                        ],
                        "opening_prefix_moves": openings[source["opening_index"]][
                            "prefix_moves"
                        ],
                        "trajectory_prefix_absolute": moves[:ply],
                        "source_path": source["source_path"],
                        "source_line": source["source_line"],
                        "source_record_sha256": source["source_record_sha256"],
                        "source_batch_sha256": source["batch_sha256"],
                        "source_action": move,
                    }
                )
                break
            if move not in legal or not game.move(move):
                raise ValueError("illegal source trajectory")
    unique = {}
    for row in eligible:
        unique.setdefault(row["state_hash"], row)
    ranked = sorted(
        unique.values(),
        key=lambda r: (sha256(f"seed560:{r['state_hash']}".encode()), r["state_hash"]),
    )
    selected, used_openings = [], set()
    for side in ("T512", "P") * 128:
        row = next(
            (
                r
                for r in ranked
                if r["source_side"] == side
                and r["opening_index"] not in used_openings
                and r not in selected
            ),
            None,
        )
        if row is not None:
            selected.append(row)
            used_openings.add(row["opening_index"])
        if len(selected) == 256:
            break
    audit = {
        "schema": "seed560-cohort-audit-v1",
        "source_commit": COMMIT,
        "archive_receipt_sha256": sha256(
            git_blob(ARCHIVE + "verification-receipt.json")
        ),
        "archive_verified_games": receipt["verified_games"],
        "eligible_first_roots": len(eligible),
        "unique_canonical_states": len(unique),
        "selected_roots": len(selected),
        "shortfall": max(0, 256 - len(selected)),
        "selected_by_source_side": {
            side: sum(r["source_side"] == side for r in selected)
            for side in ("P", "T512")
        },
        "ranking": "SHA256(seed560:<canonical_state_hash>), then canonical state hash",
        "canonical_hash": "sorted-key compact JSON SHA256",
    }
    return selected, audit


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _search(evaluator, state, simulations, seed, native):
    return arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=1.25,
        search_options=OPTIONS,
        exact_root_solve_threshold=16,
        exact_solve_stone_threshold=None,
        endgame_tablebase=native,
    )


def freeze(output: Path = OUT) -> dict:
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    states, audit = candidates()
    output.mkdir(parents=True)
    _write(
        output / "cohort-manifest.json",
        {"schema": "seed560-cohort-v1", "seed": 560, "states": states},
    )
    _write(output / "selection-audit.json", audit)
    if len(states) != 256 or audit["selected_by_source_side"] != {
        "P": 128,
        "T512": 128,
    }:
        _write(
            output / "decision.json",
            {
                "decision": "no_decision",
                "reason": "cohort_shortfall_or_imbalance",
                "shortfall": audit["shortfall"],
                "selected": len(states),
            },
        )
        return audit
    artifacts = {}
    for name, directory in (("P", P_ARTIFACT), ("initializer", INITIALIZER)):
        artifacts[name] = {
            p.name: sha256(p.read_bytes())
            for p in sorted(directory.iterdir())
            if p.is_file()
        }
    freeze_doc = {
        "schema": "seed560-execution-freeze-v1",
        "cohort_sha256": sha256((output / "cohort-manifest.json").read_bytes()),
        "selection_audit_sha256": sha256(
            (output / "selection-audit.json").read_bytes()
        ),
        "source_commit": COMMIT,
        "archive_receipt_sha256": audit["archive_receipt_sha256"],
        "artifacts": artifacts,
        "serving_model": "#556 P-E4 policy + initializer composition",
        "exact_root_assets": {
            "native_sha256": sha256(NATIVE.read_bytes()),
            "tablebase_sha256": sha256(TABLEBASE.read_bytes()),
        },
        "root_search": {
            "budgets": [128, 512],
            "c_puct": 1.25,
            "fresh_tree": True,
            "matched_seed": "int(SHA256('seed560:<state hash>:root')[:8], 16) mod 2^31; shared by 128 and 512 fresh-tree searches",
            "exact_root_cutoff": 16,
            "exact_leaf_solving": False,
            "options": OPTIONS,
        },
        "continuation": {
            "policy": "same P composition plays both seats",
            "simulations": 128,
            "exact_root_cutoff": 16,
            "exact_leaf_solving": False,
            "seed_schedule": "int(SHA256('seed560:continuation:<opening_index>:<ply>')[:8], 16) mod 2^31; same per opening and ply in both branches",
            "extra_turns": "honor KalahGame current_player",
        },
        "analysis": {
            "primary": "mean(score_512-score_128), all 256 roots",
            "score": {"win": 1, "draw": 0.5, "loss": 0},
            "opening_bootstrap_resamples": 20000,
            "bootstrap_seed": 560,
        },
        "runtime": {
            "search_workers_per_position": 1,
            "execution": "serial deterministic per-root evidence",
            "platform": __import__("platform").platform(),
            "python": __import__("platform").python_version(),
            "cpu_affinity": sorted(__import__("os").sched_getaffinity(0)),
        },
        "source_sha256": {
            relative: sha256((ROOT / relative).read_bytes())
            for relative in (
                "ml/alphazero_lite/seed560_paired_first_action.py",
                "ml/alphazero_lite/seed557_retrospective.py",
                "ml/alphazero_lite/arena.py",
                "ml/alphazero_lite/kalah_rules.py",
                "ml/alphazero_lite/native_exact_root_tablebase.py",
            )
        },
        "scope": "retrospective; continuation score specific to fixed P policy; no serving change or full-game strength claim",
    }
    _write(output / "execution-freeze.json", freeze_doc)
    return audit


def run(output: Path = OUT) -> None:
    manifest = json.loads((output / "cohort-manifest.json").read_text())
    freeze_doc = json.loads((output / "execution-freeze.json").read_text())
    if (
        sha256((output / "cohort-manifest.json").read_bytes())
        != freeze_doc["cohort_sha256"]
    ):
        raise ValueError("cohort freeze mismatch")
    if any(
        sha256((ROOT / relative).read_bytes()) != digest
        for relative, digest in freeze_doc["source_sha256"].items()
    ):
        raise ValueError("frozen_execution_source_mismatch")
    for name, directory in (("P", P_ARTIFACT), ("initializer", INITIALIZER)):
        observed = {
            p.name: sha256(p.read_bytes())
            for p in sorted(directory.iterdir())
            if p.is_file()
        }
        if observed != freeze_doc["artifacts"][name]:
            raise ValueError(f"frozen_serving_artifact_mismatch:{name}")
    if freeze_doc["exact_root_assets"] != {
        "native_sha256": sha256(NATIVE.read_bytes()),
        "tablebase_sha256": sha256(TABLEBASE.read_bytes()),
    }:
        raise ValueError("frozen_exact_root_asset_mismatch")
    evaluator = arena.ComposedArtifactEvaluator(
        arena.ArtifactEvaluator(P_ARTIFACT),
        arena.ArtifactEvaluator(INITIALIZER),
        policy_source="current",
        value_source="candidate",
    )
    roots_path, outcomes_path = (
        output / "root-searches.jsonl",
        output / "continuations.jsonl",
    )
    roots = (
        [json.loads(x) for x in roots_path.read_text().splitlines()]
        if roots_path.exists()
        else []
    )
    outcomes = (
        [json.loads(x) for x in outcomes_path.read_text().splitlines()]
        if outcomes_path.exists()
        else []
    )
    done = {r["opening_index"] for r in roots}
    with NativeExactRootTablebase(NATIVE, TABLEBASE, warm_on_start=True) as native:
        for i, row in enumerate(manifest["states"]):
            if i in done:
                continue
            seed = int(sha256(f"seed560:{row['state_hash']}:root".encode())[:8], 16) % (
                2**31
            )
            results, durations = {}, {}
            for budget in (128, 512):
                started = time.perf_counter()
                results[budget] = _search(evaluator, row["state"], budget, seed, native)
                durations[str(budget)] = time.perf_counter() - started
            actions = {n: int(results[n]["selected_move"]) for n in results}
            root_record = {
                "opening_index": i,
                "state_hash": row["state_hash"],
                "seed": seed,
                "actions": {str(n): actions[n] for n in actions},
                "latency_seconds": durations,
            }
            by_action = {}
            for budget in (128, 512):
                action = actions[budget]
                if action not in by_action:
                    by_action[action] = _continue(evaluator, row, i, action, native)
                outcomes.append(
                    {
                        "opening_index": i,
                        "state_hash": row["state_hash"],
                        "budget": budget,
                        "action": action,
                        "outcome": by_action[action],
                    }
                )
            roots.append(root_record)
            roots.sort(key=lambda x: x["opening_index"])
            roots_path.write_text(
                "".join(json.dumps(x, sort_keys=True) + "\n" for x in roots)
            )
            outcomes_path.write_text(
                "".join(json.dumps(x, sort_keys=True) + "\n" for x in outcomes)
            )
    analyze(output)


def _continue(evaluator, root_row, index, forced, native):
    game = KalahGame.from_state(root_row["state"])
    root_player = game.current_player
    if forced not in game.possible_moves() or not game.move(game.pit_index(forced)):
        raise ValueError("invalid forced first action")
    trajectory = []
    for ply in range(1000):
        if game.over():
            break
        state = game.to_state()
        seed = int(sha256(f"seed560:continuation:{index}:{ply}".encode())[:8], 16) % (
            2**31
        )
        result = _search(evaluator, state, 128, seed, native)
        action = int(result["selected_move"])
        before_player = game.current_player
        if action not in game.possible_moves() or not game.move(game.pit_index(action)):
            raise ValueError("illegal continuation action")
        exact = result.get("exact_root_decision") or {}
        trajectory.append(
            {
                "ply": ply,
                "actor": before_player,
                "action": action,
                "seed": seed,
                "state_hash": canonical_hash(state),
                "state_after": game.to_state(),
                "simulations": 128,
                "exact_root_cutoff": 16,
                "exact_root_solver_calls": int(exact.get("solver_calls", 0)),
                "exact_leaf_solving": False,
            }
        )
    else:
        raise ValueError("continuation ply cap")
    margin = game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    return {
        "score": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "margin": margin,
        "root_player": root_player,
        "trajectory": trajectory,
        "final_stores": list(game.captured_seeds),
    }


def analyze(output: Path = OUT, *, persist: bool = True) -> dict:
    roots = [
        json.loads(x) for x in (output / "root-searches.jsonl").read_text().splitlines()
    ]
    rows = [
        json.loads(x) for x in (output / "continuations.jsonl").read_text().splitlines()
    ]
    index = {(r["opening_index"], r["budget"]): r for r in rows}
    if len(roots) != 256 or len(index) != 512:
        raise ValueError("missing_or_duplicate_continuations")
    diffs = [
        index[(i, 512)]["outcome"]["score"] - index[(i, 128)]["outcome"]["score"]
        for i in range(256)
    ]
    rng = random.Random(560)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=256)) for _ in range(20000))
    changed = [
        i for i, r in enumerate(roots) if r["actions"]["128"] != r["actions"]["512"]
    ]
    result = {
        "n_openings": 256,
        "mean_score_difference_512_minus_128": statistics.fmean(diffs),
        "paired_opening_bootstrap_95_ci": [boots[499], boots[19499]],
        "bootstrap_resamples": 20000,
        "bootstrap_seed": 560,
        "action_change_rate": len(changed) / 256,
        "changed_action_roots": len(changed),
        "changed_action_score_difference": statistics.fmean(diffs[i] for i in changed)
        if changed
        else None,
        "incremental_root_search_latency_seconds_mean": statistics.fmean(
            r["latency_seconds"]["512"] - r["latency_seconds"]["128"] for r in roots
        ),
        "decision": "propose_separate_selective_budget_rule_test"
        if statistics.fmean(diffs) >= 0.03 and boots[499] > 0
        else "no_selective_budget_rule_test",
        "scope": "retrospective; fixed P continuation policy only; no full-game strength or serving authorization",
    }
    if persist:
        _write(output / "analysis.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--analyze", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.out)
            if args.freeze
            else analyze(args.out)
            if args.analyze
            else run(args.out),
            sort_keys=True,
            indent=2,
        )
    )
