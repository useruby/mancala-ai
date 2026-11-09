"""Retrospective paired first-action budget diagnostic from the #556 archive."""

from __future__ import annotations

import json
import hashlib
import os
import random
import statistics
import time
from pathlib import Path

from ml.alphazero_lite import arena, seed552_c108_serving_p as serving_p
from ml.alphazero_lite import seed555_policy_strength_screen as seed555
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
RUN0004_COHORT_SHA256 = (
    "fe24d68aea9a158cb441aa886a60adc26631805c5b1213f14fceb4aa4ce4a254"
)
RUN0004_RUNNER_SHA256 = (
    "70b07ad5e9af020606f83dc45fe7e603b04d8aa3167fcd1eba0fc581ab3b9bf7"
)
P_ARTIFACT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4"
)
INITIALIZER = (
    ROOT
    / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/initializer"
)
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
RUN_COMMITS = "root-commits"
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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        parent_fd = os.open(path.parent.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        stream.write((json.dumps(value, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rows_by_index(rows: list[dict], label: str) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        index = int(row["opening_index"])
        if index in indexed:
            raise ValueError(f"duplicate_{label}_opening:{index}")
        indexed[index] = row
    return indexed


def _read_resume_evidence(output: Path) -> tuple[list[dict], list[dict], set[int]]:
    """Load complete root+two-arm commits, rejecting partial legacy ledgers."""
    root_path, outcomes_path = (
        output / "root-searches.jsonl",
        output / "continuations.jsonl",
    )

    def read_jsonl(path: Path) -> list[dict]:
        return (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.exists()
            else []
        )

    existing_roots = read_jsonl(root_path)
    existing_outcomes = read_jsonl(outcomes_path)
    commits_dir = output / RUN_COMMITS
    if not commits_dir.exists():
        roots = _rows_by_index(existing_roots, "root")
        outcomes = {}
        for row in existing_outcomes:
            key = (int(row["opening_index"]), int(row["budget"]))
            if key in outcomes:
                raise ValueError(f"duplicate_continuation_arm:{key}")
            outcomes[key] = row
        indices = set(roots) | {key[0] for key in outcomes}
        for index in indices:
            if index not in roots or any(
                (index, budget) not in outcomes for budget in (128, 512)
            ):
                raise ValueError(f"incomplete_saved_root_bundle:{index}")
            for budget in (128, 512):
                row = outcomes[(index, budget)]
                if row.get("state_hash") != roots[index].get("state_hash") or row.get(
                    "action"
                ) != roots[index].get("actions", {}).get(str(budget)):
                    raise ValueError(
                        f"contradictory_saved_root_bundle:{index}:{budget}"
                    )
        if len(outcomes) != 2 * len(roots):
            raise ValueError("orphan_saved_continuation_arm")
        if len(roots) > 256 or set(roots) != set(range(len(roots))):
            raise ValueError("legacy_root_records_are_not_a_frozen_cohort_prefix")
        # Keep the fully published run-0004 byte-identical. A complete legacy
        # prefix that will be extended is migrated before the first new search.
        if roots and len(roots) < 256:
            for index in sorted(roots):
                _commit_root_bundle(
                    output,
                    roots[index],
                    [outcomes[(index, budget)] for budget in (128, 512)],
                )
            return _read_resume_evidence(output)
        return (
            sorted(roots.values(), key=lambda row: row["opening_index"]),
            sorted(
                outcomes.values(), key=lambda row: (row["opening_index"], row["budget"])
            ),
            set(roots),
        )

    committed = {}
    for path in sorted(commits_dir.glob("opening-*.json")):
        bundle = json.loads(path.read_text())
        index = int(bundle["opening_index"])
        if path.name != f"opening-{index:04d}.json" or index in committed:
            raise ValueError(f"invalid_or_duplicate_root_commit:{path.name}")
        root = bundle.get("root")
        arms = bundle.get("arms")
        if (
            not isinstance(root, dict)
            or root.get("opening_index") != index
            or not isinstance(arms, list)
            or {int(row.get("budget", -1)) for row in arms} != {128, 512}
            or len(arms) != 2
        ):
            raise ValueError(f"incomplete_root_commit_record:{index}")
        by_budget = {int(row["budget"]): row for row in arms}
        for budget, row in by_budget.items():
            if (
                row.get("opening_index") != index
                or row.get("state_hash") != root.get("state_hash")
                or row.get("action") != root.get("actions", {}).get(str(budget))
            ):
                raise ValueError(f"contradictory_root_commit_record:{index}:{budget}")
        committed[index] = (root, by_budget)

    # Validate the entire aggregate tail and its relation to existing commits
    # before the first migration write. This keeps rejection side-effect free.
    aggregate_roots = _rows_by_index(existing_roots, "aggregate_root")
    aggregate_arms = {}
    for row in existing_outcomes:
        key = (int(row["opening_index"]), int(row["budget"]))
        if key[1] not in (128, 512):
            raise ValueError(f"invalid_aggregate_arm_budget:{key}")
        if key in aggregate_arms:
            raise ValueError(f"duplicate_aggregate_arm:{key}")
        aggregate_arms[key] = row
    aggregate_indices = set(aggregate_roots) | {key[0] for key in aggregate_arms}
    legacy_to_migrate = []
    for index in sorted(aggregate_indices - set(committed)):
        root = aggregate_roots.get(index)
        if root is None or any(
            (index, budget) not in aggregate_arms for budget in (128, 512)
        ):
            raise ValueError(f"incomplete_legacy_root_bundle:{index}")
        arms = {budget: aggregate_arms[(index, budget)] for budget in (128, 512)}
        for budget, row in arms.items():
            if row.get("state_hash") != root.get("state_hash") or row.get(
                "action"
            ) != root.get("actions", {}).get(str(budget)):
                raise ValueError(f"contradictory_legacy_root_bundle:{index}:{budget}")
        legacy_to_migrate.append((index, root, arms))

    # Existing rows for committed roots may be a prefix of their atomic
    # aggregate rewrite, but any extant disagreement is contradictory.
    for index in sorted(aggregate_indices & set(committed)):
        committed_root, committed_arms = committed[index]
        root = aggregate_roots.get(index)
        if root is not None and root != committed_root:
            raise ValueError(f"aggregate_root_conflicts_with_commit:{index}")
        for budget in (128, 512):
            row = aggregate_arms.get((index, budget))
            if row is not None and row != committed_arms[budget]:
                raise ValueError(
                    f"aggregate_arm_conflicts_with_commit:{index}:{budget}"
                )

    # The entire tail is known complete and consistent; only now write bundles.
    for index, root, arms in legacy_to_migrate:
        _commit_root_bundle(output, root, [arms[128], arms[512]])
        committed[index] = (root, arms)

    # Existing aggregate ledgers may be torn between atomic replacements.
    # They are recoverable only when every extant row agrees with a commit.
    expected_roots = {index: item[0] for index, item in committed.items()}
    expected_arms = {
        (index, budget): row
        for index, (_, arms) in committed.items()
        for budget, row in arms.items()
    }
    for index, row in aggregate_roots.items():
        if expected_roots.get(index) != row:
            raise ValueError(f"aggregate_root_conflicts_with_commit:{index}")
    for key, row in aggregate_arms.items():
        if expected_arms.get(key) != row:
            raise ValueError(f"aggregate_arm_conflicts_with_commit:{key}")
    roots = [expected_roots[index] for index in sorted(expected_roots)]
    outcomes = [expected_arms[key] for key in sorted(expected_arms)]
    if existing_roots != roots:
        _atomic_jsonl(root_path, roots)
    if existing_outcomes != outcomes:
        _atomic_jsonl(outcomes_path, outcomes)
    return roots, outcomes, set(committed)


def _commit_root_bundle(output: Path, root: dict, arms: list[dict]) -> None:
    index = int(root["opening_index"])
    path = output / RUN_COMMITS / f"opening-{index:04d}.json"
    bundle = {"opening_index": index, "root": root, "arms": arms}
    if path.exists():
        if json.loads(path.read_text()) != bundle:
            raise ValueError(f"root_commit_conflict:{index}")
        return
    _atomic_json(path, bundle)


def authenticated_inputs(
    artifact_dirs: dict[str, Path] | None = None,
    *,
    require_materialized_tablebase: bool = False,
) -> dict:
    """Authenticate the exact evaluator and solver files against #556/#552/#555."""
    from ml.alphazero_lite.seed557_retrospective import git_blob as authenticated_blob

    artifact_dirs = artifact_dirs or {"P": P_ARTIFACT, "initializer": INITIALIZER}
    registration = json.loads(authenticated_blob(ARCHIVE + "registration.json"))
    preflight = json.loads(
        authenticated_blob(ARCHIVE + "runtime-policy-preflight.json")
    )
    receipt = json.loads(authenticated_blob(ARCHIVE + "verification.json"))
    if (
        receipt.get("status") != "verified_complete_with_completion_order_correction"
        or not receipt.get("component_artifacts_rehashed")
        or not receipt.get("native_probe_rehashed")
        or not receipt.get("repository_lfs_pointer_rehashed")
        or registration.get("components", {}).get("P-E4") is None
    ):
        raise ValueError("authenticated_seed556_publication_missing")
    expected_components = {
        "P": (
            "P-E4",
            serving_p.P_SHA256,
        ),
        "initializer": (
            "initializer",
            seed555.INITIALIZER_SHA256,
        ),
    }
    result = {}
    if set(artifact_dirs) != set(expected_components):
        raise ValueError("evaluator_component_set_mismatch")
    for name, (registered_name, expected_model_hash) in expected_components.items():
        expected_files = registration["components"][registered_name]
        directory = artifact_dirs[name]
        actual_files = {
            path.name: _file_sha(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        if (
            actual_files != expected_files
            or actual_files.get("model.npz") != expected_model_hash
        ):
            raise ValueError(f"unauthenticated_serving_artifact:{name}")
        result[name] = actual_files

    native_hash = _file_sha(NATIVE)
    expected_native_hash = preflight.get("exact_root_native_probe_sha256")
    expected_tablebase_hash = preflight.get("exact_root_tablebase_sha256")
    if (
        native_hash != expected_native_hash
        or expected_tablebase_hash != seed555.TABLEBASE_SHA256
        or receipt.get("materialized_tablebase_sha256") != expected_tablebase_hash
        or receipt.get("materialized_tablebase_size_bytes") != seed555.TABLEBASE_SIZE
    ):
        raise ValueError("unauthenticated_seed556_exact_root_assets")
    with TABLEBASE.open("rb") as stream:
        pointer_or_asset = stream.read(256)
    lfs_oid = None
    if pointer_or_asset.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        for line in pointer_or_asset.decode("ascii").splitlines()[1:]:
            if line.startswith("oid sha256:"):
                lfs_oid = line.removeprefix("oid sha256:")
                break
    if lfs_oid is not None:
        if lfs_oid != expected_tablebase_hash:
            raise ValueError("tablebase_lfs_object_identity_mismatch")
        if require_materialized_tablebase:
            raise ValueError("exact_root_tablebase_not_materialized")
    elif (
        _file_sha(TABLEBASE) != expected_tablebase_hash
        or TABLEBASE.stat().st_size != seed555.TABLEBASE_SIZE
    ):
        raise ValueError("materialized_tablebase_identity_mismatch")
    return {
        "components": result,
        "native_probe_sha256": native_hash,
        "tablebase_sha256": expected_tablebase_hash,
        "tablebase_size_bytes": seed555.TABLEBASE_SIZE,
    }


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
    authenticated = authenticated_inputs(require_materialized_tablebase=True)
    artifacts = authenticated["components"]
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
            "native_sha256": authenticated["native_probe_sha256"],
            "tablebase_sha256": authenticated["tablebase_sha256"],
            "tablebase_size_bytes": authenticated["tablebase_size_bytes"],
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
    runner_relative = "ml/alphazero_lite/seed560_paired_first_action.py"
    is_preserved_run0004 = (
        output.resolve() == OUT.resolve()
        and freeze_doc.get("cohort_sha256") == RUN0004_COHORT_SHA256
        and freeze_doc.get("source_sha256", {}).get(runner_relative)
        == RUN0004_RUNNER_SHA256
    )
    if any(
        sha256((ROOT / relative).read_bytes()) != digest
        for relative, digest in freeze_doc["source_sha256"].items()
        if relative != runner_relative
    ) or (
        not is_preserved_run0004
        and sha256((ROOT / runner_relative).read_bytes())
        != freeze_doc["source_sha256"].get(runner_relative)
    ):
        raise ValueError("frozen_execution_source_mismatch")
    authenticated = authenticated_inputs(require_materialized_tablebase=True)
    if (
        authenticated["components"] != freeze_doc["artifacts"]
        or authenticated["native_probe_sha256"]
        != freeze_doc["exact_root_assets"]["native_sha256"]
        or authenticated["tablebase_sha256"]
        != freeze_doc["exact_root_assets"]["tablebase_sha256"]
        or authenticated["tablebase_size_bytes"]
        != freeze_doc["exact_root_assets"]["tablebase_size_bytes"]
    ):
        raise ValueError("frozen_exact_root_asset_mismatch")
    roots, outcomes, done = _read_resume_evidence(output)
    for root in roots:
        index = int(root["opening_index"])
        if (
            not 0 <= index < len(manifest["states"])
            or root.get("state_hash") != manifest["states"][index]["state_hash"]
        ):
            raise ValueError(f"saved_root_does_not_match_frozen_cohort:{index}")
    evaluator = arena.ComposedArtifactEvaluator(
        arena.ArtifactEvaluator(P_ARTIFACT),
        arena.ArtifactEvaluator(INITIALIZER),
        policy_source="current",
        value_source="candidate",
    )
    if authenticated_inputs() != authenticated:
        raise ValueError("serving_artifact_changed_during_evaluator_load")
    roots_path, outcomes_path = (
        output / "root-searches.jsonl",
        output / "continuations.jsonl",
    )
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
            arm_records = []
            for budget in (128, 512):
                action = actions[budget]
                if action not in by_action:
                    by_action[action] = _continue(evaluator, row, i, action, native)
                arm_records.append(
                    {
                        "opening_index": i,
                        "state_hash": row["state_hash"],
                        "budget": budget,
                        "action": action,
                        "outcome": by_action[action],
                    }
                )
            _commit_root_bundle(output, root_record, arm_records)
            roots.append(root_record)
            roots.sort(key=lambda x: x["opening_index"])
            outcomes.extend(arm_records)
            outcomes.sort(key=lambda x: (x["opening_index"], x["budget"]))
            _atomic_jsonl(roots_path, roots)
            _atomic_jsonl(outcomes_path, outcomes)
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
