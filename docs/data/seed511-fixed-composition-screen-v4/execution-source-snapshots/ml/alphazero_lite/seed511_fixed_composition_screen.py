"""Exploratory seed511 screen of the fixed E4-policy/initializer-value model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed488_strength_evaluation as base
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed511-fixed-composition-screen-v4")
SEED = 511
OPENINGS = 1024
GAMES = 2048
RESAMPLES = 20_000
E4_SHA = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
RUNTIME_RECOVERY_COMMIT = "71fa5248cff5da5e3e7aa46f6d71057ee3e9ab5c"
RUNTIME_RECOVERY_PATHS = {
    "exact_root_native_probe": "model-artifact/runtime/kalah_v1_tablebase",
    "exact_root_tablebase": "model-artifact/runtime/kalah_v1_21.kvtb",
}
SOURCE_EXCLUSIONS = (
    "docs/data/seed482-exclusion-manifest.json.gz",
    "docs/data/seed483-historical-probe-identities.json.gz",
    "docs/data/seed484-extended-probe-union.json.gz",
)
REPLAY_SUITES = {
    "seed488": "docs/data/seed488-frozen-strength-evaluation",
    "seed499": "docs/data/seed499-budget-comparison",
    "seed505": "docs/data/seed505-strength-screen-v3",
    "seed508": "docs/data/seed508-policy-value-crossover",
}
SEED500 = "docs/data/seed500-selfplay-512"
PILOT500 = "docs/data/seed500-selfplay-64.jsonl"
INITIAL_STATE = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
RUNTIME_SOURCES = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/classic_mcts.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/seed488_strength_evaluation.py",
    "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
    "ml/alphazero_lite/seed508_policy_value_crossover.py",
    "ml/alphazero_lite/verify_seed508_policy_value_crossover.py",
    "ml/alphazero_lite/verify_seed505_strength_screen.py",
    "ml/alphazero_lite/seed482_fresh_confirmation.py",
    "ml/alphazero_lite/seed511_fixed_composition_screen.py",
    "ml/alphazero_lite/verify_seed511_fixed_composition_screen.py",
    "ml/alphazero_lite/test_seed511_fixed_composition_screen.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _step_and_record(game: KalahGame, absolute_move: int, identities: dict[str, set[str]]) -> None:
    if game.over() or absolute_move // 6 != game.current_player:
        raise ValueError("historical_replay_illegal_action")
    relative = absolute_move - game.current_player * 6
    if relative not in game.possible_moves():
        raise ValueError("historical_replay_illegal_action")
    state = game.to_state()
    identities["canonical"].add(canonical_identity(state))
    identities["float32"].add(float32_identity(state))
    if not game.move(absolute_move):
        raise ValueError("historical_replay_move_rejected")


def replay_prefix_and_trajectory(
    prefix: list[int], trajectory: list[int], identities: dict[str, set[str]]
) -> None:
    game = KalahGame.from_state(INITIAL_STATE)
    for action in prefix:
        if game.over() or int(action) not in game.possible_moves():
            raise ValueError("historical_opening_prefix_illegal")
        _step_and_record(game, game.pit_index(int(action)), identities)
    for action in trajectory:
        _step_and_record(game, int(action), identities)
    if not game.over() and len(trajectory) < 200:
        raise ValueError("historical_game_not_terminal")


def _replay_outcomes(path: Path, identities: dict[str, set[str]]) -> int:
    rows = read_rows(path)
    for row in rows:
        prefix = row.get("opening_prefix_moves")
        trajectory = row.get("trajectory")
        if not isinstance(prefix, list) or not isinstance(trajectory, str):
            raise ValueError(f"historical_game_replay_fields:{path}")
        replay_prefix_and_trajectory(
            [int(x) for x in prefix],
            [int(x) for x in trajectory.split(",") if x],
            identities,
        )
    return len(rows)


def _replay_seed508(directory: Path, identities: dict[str, set[str]]) -> int:
    count = 0
    batches = directory / "batches"
    for games_path in sorted(batches.glob("*-games-*.jsonl")):
        arm, _, stem_index = games_path.name.split("-")
        index = int(stem_index.split(".")[0])
        opening_path = batches / f"{arm}-opening-{index:04d}.jsonl"
        if not opening_path.is_file():
            raise ValueError(f"seed508_opening_batch_missing:{arm}:{index}")
        opening_rows = read_rows(opening_path)
        if len(opening_rows) != 1:
            raise ValueError(f"seed508_opening_batch_shape:{arm}:{index}")
        prefix = opening_rows[0]["prefix_moves"]
        games = read_rows(games_path)
        if len(games) != 2:
            raise ValueError(f"seed508_game_batch_shape:{arm}:{index}")
        for game in games:
            replay_prefix_and_trajectory(
                [int(x) for x in prefix],
                [int(x) for x in game["trajectory"].split(",") if x],
                identities,
            )
        count += len(games)
    if count != 2048:
        raise ValueError(f"seed508_replay_count:{count}")
    return count


def _seed500_sources(root: Path) -> list[Path]:
    corpus = root / SEED500
    accounting = json.loads((corpus / "accounting.json").read_text())
    paths = [root / entry["path"] for entry in accounting["shards"]]
    if len(paths) != 8:
        raise ValueError("seed500_shard_count")
    for entry, path in zip(accounting["shards"], paths, strict=True):
        if not path.is_file() or sha(path) != entry["corpus_sha256"]:
            raise ValueError(f"seed500_committed_shard_hash:{path}")
    return paths


def reconstruct_exclusions(root: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Rebuild the registered union and replay all specified committed games."""
    root = root.resolve()
    historic_pairs = base.reconstructable_exclusions(root)
    identities = {
        "canonical": {pair[0] for pair in historic_pairs},
        "float32": {pair[1] for pair in historic_pairs},
    }
    accounting: dict[str, Any] = {
        "reconstructable_historical_union": {
            "pairs": len(historic_pairs),
            "canonical_identities": len(identities["canonical"]),
            "float32_identities": len(identities["float32"]),
            "sources": list(SOURCE_EXCLUSIONS),
        },
        "replayed_sources": {},
    }

    def record_delta(name: str, games: int | None = None, rows: int | None = None) -> None:
        source = accounting["replayed_sources"].setdefault(name, {})
        if games is not None:
            source["games_replayed"] = games
        if rows is not None:
            source["source_rows"] = rows
        source["new_canonical_identities"] = len(identities["canonical"] - prior_canonical)
        source["new_float32_identities"] = len(identities["float32"] - prior_float32)
        source["cumulative_canonical_identities"] = len(identities["canonical"])
        source["cumulative_float32_identities"] = len(identities["float32"])

    for key in ("seed488", "seed499", "seed505"):
        directory = root / REPLAY_SUITES[key]
        prior_canonical = set(identities["canonical"])
        prior_float32 = set(identities["float32"])
        n = _replay_outcomes(directory / "outcomes.jsonl", identities)
        record_delta(key, games=n)
    prior_canonical = set(identities["canonical"])
    prior_float32 = set(identities["float32"])
    seed508_count = _replay_seed508(root / REPLAY_SUITES["seed508"], identities)
    accounting["replayed_sources"]["seed508"] = {
        "games_replayed": seed508_count,
        "new_canonical_identities": len(identities["canonical"] - prior_canonical),
        "new_float32_identities": len(identities["float32"] - prior_float32),
        "cumulative_canonical_identities": len(identities["canonical"]),
        "cumulative_float32_identities": len(identities["float32"]),
    }
    prior_canonical = set(identities["canonical"])
    prior_float32 = set(identities["float32"])
    seed500_paths = (root / PILOT500, *_seed500_sources(root))
    seed500_rows = 0
    for path in seed500_paths:
        for row in read_rows(path):
            seed500_rows += 1
            encoded = np.asarray(row["state"], dtype="<f4")
            if encoded.shape != (27,) or not np.isfinite(encoded).all():
                raise ValueError(f"seed500_input_shape:{path}")
            # Reconstruct canonical rule state from the validated, production-f32 input.
            from ml.alphazero_lite.seed482_fresh_confirmation import state_from_encoded

            state = state_from_encoded(encoded)
            identities["canonical"].add(canonical_identity(state))
            identities["float32"].add(float32_identity(state))
    accounting["replayed_sources"]["seed500-pilot-and-eight-shards"] = {
        "source_rows": seed500_rows,
        "new_canonical_identities": len(identities["canonical"] - prior_canonical),
        "new_float32_identities": len(identities["float32"] - prior_float32),
        "cumulative_canonical_identities": len(identities["canonical"]),
        "cumulative_float32_identities": len(identities["float32"]),
        "archive_validation": "complete; per-shard hashes matched accounting.json; replay and production-loader accounting verified",
    }
    accounting["union_after_sources"] = {
        "canonical_identities": len(identities["canonical"]),
        "float32_identities": len(identities["float32"]),
    }
    accounting["seed461"] = "historical reconstructable set retains its disclosed incomplete replay gap; not reopened"
    return identities, accounting


def make_suite(identities: dict[str, set[str]], seed: int = SEED) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate uniformly random rules-only prefixes using seed488's 4–12 ply law."""
    rng = random.Random(seed)
    chosen_canonical: set[str] = set()
    chosen_float32: set[str] = set()
    openings: list[dict[str, Any]] = []
    rejected = {"historical_identity": 0, "duplicate_selected_identity": 0, "terminal_prefix": 0}
    attempts = 0
    while len(openings) < OPENINGS and attempts < 2_000_000:
        attempts += 1
        game = KalahGame.from_state(INITIAL_STATE)
        absolute: list[int] = []
        target = rng.randint(4, 12)
        for _ in range(target):
            if game.over():
                break
            move = game.pit_index(rng.choice(game.possible_moves()))
            absolute.append(move)
            if not game.move(move):
                raise ValueError("generated_prefix_move")
        if game.over():
            rejected["terminal_prefix"] += 1
            continue
        state = game.to_state()
        can, f32 = canonical_identity(state), float32_identity(state)
        if can in identities["canonical"] or f32 in identities["float32"]:
            rejected["historical_identity"] += 1
            continue
        if can in chosen_canonical or f32 in chosen_float32:
            rejected["duplicate_selected_identity"] += 1
            continue
        replay = KalahGame.from_state(INITIAL_STATE)
        relative = []
        for absolute_move in absolute:
            relative.append(absolute_move - replay.current_player * 6)
            replay.move(absolute_move)
        if replay.to_state() != state:
            raise ValueError("generated_prefix_replay")
        chosen_canonical.add(can)
        chosen_float32.add(f32)
        opening = {
            "opening_index": len(openings),
            "opening_contract": "arena_player_relative_v2",
            "prefix_moves": relative,
            "source_prefix_moves_absolute": absolute,
            "state": state,
            "state_hash": hashlib.sha256(canonical_json(state)).hexdigest(),
        }
        openings.append(opening)
    if len(openings) != OPENINGS:
        raise ValueError(f"seed511_opening_shortfall:{len(openings)}")
    rejected["attempts"] = attempts
    return openings, rejected


def suite_bytes(openings: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in openings)


def component_manifest(root: Path) -> dict[str, Any]:
    reg_path = root / "docs/data/seed508-policy-value-crossover/registration.json"
    prior = json.loads(reg_path.read_text())
    components = {}
    for name in ("E4", "initializer"):
        source = root / "docs/data/seed508-policy-value-crossover/runtime-artifacts" / name
        frozen = prior["components"][name]
        files = {f: sha(source / f) for f in frozen["files"]}
        if files != frozen["files"]:
            raise ValueError(f"bound_component_artifact_mismatch:{name}")
        components[name] = {
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "checkpoint_source": (
                "docs/data/seed504/seed504_epoch4.npz"
                if name == "E4"
                else "docs/data/seed508-policy-value-crossover/initializer.npz"
            ),
            "artifact_source": str(source.relative_to(root)),
            "files": files,
        }
        if sha(root / components[name]["checkpoint_source"]) != frozen["checkpoint_sha256"]:
            raise ValueError(f"bound_checkpoint_mismatch:{name}")
        if name == "E4" and files["model.npz"] != E4_SHA:
            raise ValueError("E4_checkpoint_hash")
        if name == "initializer" and files["model.npz"] != INIT_SHA:
            raise ValueError("initializer_checkpoint_hash")
    return components


def exclusion_source_paths(root: Path) -> list[str]:
    paths = [*SOURCE_EXCLUSIONS, PILOT500, f"{SEED500}/accounting.json"]
    paths.extend(f"{SEED500}/shard-50300{i}.jsonl" for i in range(8))
    for key in ("seed488", "seed499", "seed505"):
        paths.extend(
            [
                f"{REPLAY_SUITES[key]}/registration.json",
                f"{REPLAY_SUITES[key]}/openings.jsonl",
                f"{REPLAY_SUITES[key]}/outcomes.jsonl",
            ]
        )
    paths.extend(
        [
            f"{REPLAY_SUITES['seed508']}/openings.jsonl",
            f"{REPLAY_SUITES['seed508']}/commands.jsonl",
            f"{REPLAY_SUITES['seed508']}/registration.json",
        ]
    )
    paths.extend(
        str(path.relative_to(root))
        for pattern in ("*-opening-*.jsonl", "*-games-*.jsonl")
        for path in sorted((root / REPLAY_SUITES["seed508"] / "batches").glob(pattern))
    )
    return paths


def dependency_versions() -> list[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def verify_frozen_inputs(root: Path, registration: dict[str, Any]) -> None:
    for rel, digest in registration["sources_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_source_changed:{rel}")
    for rel, digest in registration["exclusion_source_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_exclusion_source_changed:{rel}")
    if dependency_versions() != registration["environment"]["pip_freeze"]:
        raise ValueError("frozen_environment_dependency_mismatch")
    seed508 = json.loads(
        (root / "docs/data/seed508-policy-value-crossover/registration.json").read_text()
    )
    if seed508["runtime"]["runtime_identities_sha256"] != registration["runtime_identity_sha256"]:
        raise ValueError("frozen_runtime_identity_manifest_mismatch")
    for name, component in registration["components"].items():
        if sha(root / component["checkpoint_source"]) != component["checkpoint_sha256"]:
            raise ValueError(f"frozen_checkpoint_changed:{name}")
        artifact = root / component["artifact"]
        if {f: sha(artifact / f) for f in component["files"]} != component["files"]:
            raise ValueError(f"frozen_component_changed:{name}")
    bound_runtime = registration["runtime_identity_sha256"]
    copied_runtime = registration["runtime_assets"]
    copied_by_source = {asset["source_path"]: asset for asset in copied_runtime.values()}
    for rel, digest in bound_runtime.items():
        if rel in copied_by_source and copied_by_source[rel].get("lfs_oid"):
            path = Path(copied_by_source[rel]["runtime_materialized_path"])
        elif rel in copied_by_source:
            path = root / copied_by_source[rel]["artifact"]
        else:
            path = root / rel
        if sha(path) != digest:
            raise ValueError(f"frozen_runtime_identity_changed:{rel}")


def recover_runtime_assets(root: Path, destination: Path, expected: dict[str, str]) -> dict[str, Any]:
    assets = {}
    for name, source_rel in RUNTIME_RECOVERY_PATHS.items():
        git_payload = subprocess.run(
            ["git", "show", f"{RUNTIME_RECOVERY_COMMIT}:{source_rel}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        if name == "exact_root_tablebase":
            pointer = git_payload.decode("utf-8").splitlines()
            oid_line = next((line for line in pointer if line.startswith("oid sha256:")), None)
            if oid_line != f"oid sha256:{expected[source_rel]}":
                raise ValueError("seed508_tablebase_lfs_pointer_identity")
            git_common = subprocess.run(
                ["git", "rev-parse", "--git-common-dir"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            lfs_root_path = Path(git_common)
            if not lfs_root_path.is_absolute():
                lfs_root_path = root / lfs_root_path
            lfs_root_path = lfs_root_path / "lfs" / "objects"
            materialized = lfs_root_path / expected[source_rel][:2] / expected[source_rel][2:4] / expected[source_rel]
            if not materialized.is_file() or sha(materialized) != expected[source_rel]:
                raise ValueError("seed508_tablebase_lfs_object_unavailable_or_corrupt")
            assets[name] = {
                "source_commit": RUNTIME_RECOVERY_COMMIT,
                "source_git_path": source_rel,
                "source_path": source_rel,
                "source_pointer_sha256": hashlib.sha256(git_payload).hexdigest(),
                "lfs_oid": expected[source_rel],
                "artifact": source_rel,
                "runtime_materialized_path": str(materialized),
                "sha256": expected[source_rel],
                "recovered_from": "shared_git_lfs_object_cache",
            }
            continue
        payload = git_payload
        recovered_from = "committed_git_blob"
        if hashlib.sha256(payload).hexdigest() != expected[source_rel]:
            raise ValueError(f"historical_runtime_blob_hash:{name}")
        digest = expected[source_rel]
        assets[name] = {
            "source_commit": RUNTIME_RECOVERY_COMMIT,
            "source_git_path": source_rel,
            "source_path": source_rel,
            "recovered_from": recovered_from,
            "git_blob_sha1": subprocess.run(
                ["git", "rev-parse", f"{RUNTIME_RECOVERY_COMMIT}:{source_rel}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "artifact": source_rel,
            "sha256": digest,
        }
    return assets


def materialize_bound_tablebase(root: Path, registration: dict[str, Any]) -> bytes | None:
    """Temporarily hard-link the authenticated LFS object at its sidecar path."""
    asset = registration["runtime_assets"]["exact_root_tablebase"]
    path = root / asset["artifact"]
    if path.is_file() and sha(path) == asset["sha256"]:
        return None
    pointer = subprocess.run(
        ["git", "show", f"HEAD:{asset['source_path']}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(pointer).hexdigest() != asset["source_pointer_sha256"]:
        raise ValueError("seed511_tablebase_pointer_changed")
    if not path.is_file() or path.read_bytes() != pointer:
        raise ValueError("seed511_tablebase_path_not_original_pointer")
    temp = path.with_name(path.name + ".seed511-materialized")
    if temp.exists():
        raise ValueError("seed511_tablebase_materialization_temp_exists")
    os.link(Path(asset["runtime_materialized_path"]), temp)
    os.replace(temp, path)
    if sha(path) != asset["sha256"]:
        raise ValueError("seed511_tablebase_materialization_hash")
    return pointer


def restore_bound_tablebase(root: Path, registration: dict[str, Any], pointer: bytes | None) -> None:
    if pointer is None:
        return
    path = root / registration["runtime_assets"]["exact_root_tablebase"]["artifact"]
    temp = path.with_name(path.name + ".seed511-restore")
    temp.write_bytes(pointer)
    os.replace(temp, path)
    if sha(path) != registration["runtime_assets"]["exact_root_tablebase"]["source_pointer_sha256"]:
        raise ValueError("seed511_tablebase_pointer_restore_hash")


def prepare(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(
        (out / name).exists()
        for name in ("registration.json", "execution-freeze.json", "openings.jsonl", "commands.jsonl", "outcomes.jsonl")
    ):
        raise ValueError("seed511_output_already_exists")
    identities, exclusion_accounting = reconstruct_exclusions(root)
    openings, rejected = make_suite(identities)
    suite = suite_bytes(openings)
    components = component_manifest(root)
    import torch

    seed508_registration = json.loads(
        (root / "docs/data/seed508-policy-value-crossover/registration.json").read_text()
    )
    dependencies = dependency_versions()
    registration = {
        "schema": "seed511-fixed-composition-screen-v1",
        "status": "frozen_before_candidate_inference_or_games",
        "interpretation": "new exploratory suite; not an untouched or fresh-confirmation holdout",
        "historical_exclusion_scope": "Uses the reconstructable historical identity set plus the explicitly named seed500 and seed488/499/505/508 replay sources below. It does not claim complete historical exclusion. The seed461 replay gap remains unresolved and was not reopened.",
        "checkout_recovery": {
            "remote": "origin git@github.com:useruby/mancala-ai.git",
            "evaluation_base_revision": "8f5063df571e521363afe4c01e6eb431aca32d3d (#510)",
            "seed503_merge_revision": "50f42ef63e72d297b66c5f9dbaf52c0bf8f9e1f6",
            "seed503_merge_in_base_history": True,
            "sparse_checkout": False,
            "lfs_materialization": "the #510 worktree had the seed508-bound 134-byte Git LFS pointer because smudge/filter configuration was disabled; the exact 709,635,072-byte object with matching f126... OID was present in the shared Git LFS cache and is temporarily hard-linked at its registered runtime path for arena execution, then the pointer is restored",
            "resolution": "isolated detached worktree at #510; initial apparent absence was caused by a non-recursive path listing, not missing committed history",
            "archive_verification": {
                "shards": 8,
                "shard_sha256_matches_accounting": True,
                "replayed_games": 512,
                "decision_rows": 18281,
                "production_loader": "accepted",
                "loader_accounting_matches": True,
                "aggregate_corpus_sha256": "a2602c6c908c41be273c357a425fe06267e230a700ecd5368df15e70b9836534",
            },
        },
        "decision": {
            "score": "(wins + 0.5 * draws) / 2048",
            "pass": "one-sided 95% paired-opening bootstrap lower bound strictly greater than 0.50",
            "otherwise": "close this fixed-composition screening branch without changing settings or selecting another checkpoint",
            "promotion": False,
            "fresh_confirmation": False,
            "seed505_failed_gate_reopened": False,
            "passing_only_permits_proposing_later_1200_simulation_test": True,
        },
        "candidate": {
            "policy_output": "seed504 E4 policy output with its own trunk",
            "value_output": "frozen initializer value output with its own trunk",
            "checkpoint_tensor_splicing": False,
            "training": False,
        },
        "opponent": "complete frozen initializer artifact",
        "components": components,
        "suite": {
            "seed": SEED,
            "openings": OPENINGS,
            "games": GAMES,
            "generation": "uniform random legal rules-only prefixes; target prefix length random.Random(511).randint(4, 12), matching seed488; reject terminal prefixes and canonical OR production-float32-input collisions; no outcome-based replacement",
            "sha256": hashlib.sha256(suite).hexdigest(),
            "rejected": rejected,
            "opening_indices": [0, OPENINGS - 1],
        },
        "exclusions": exclusion_accounting,
        "runtime": {
            "simulations_per_side": 128,
            "c_puct": 1.25,
            "workers": 1,
            "seed_contract": "azlite_eval_seed_v2; base seed = 511 + opening index",
            "max_plies": 200,
            "random_opening_plies": 0,
            "exact_root": "same seed488/508 registered exact-root behavior; threshold 16, final_score_margin objective, highest legal network prior then lowest move index tie rule, exact leaf solve disabled",
            "search_options": json.loads((root / "docs/data/seed508-policy-value-crossover/registration.json").read_text())["runtime"]["search_options"],
        },
        "analysis": {
            "primary": "P standalone score",
            "paired_cluster": "opening; average both seats within opening",
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": SEED,
            "one_sided_lower_quantile": 0.05,
            "descriptive_two_sided_quantiles": [0.025, 0.975],
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": dependencies,
            "pip_freeze_sha256": hashlib.sha256(canonical_json(dependencies)).hexdigest(),
        },
        "runtime_identity_sha256": seed508_registration["runtime"]["runtime_identities_sha256"],
        "runtime_asset_recovery": {
            "source_commit": RUNTIME_RECOVERY_COMMIT,
            "reason": "the seed508-bound tablebase was present as a Git LFS pointer but not materialized in the #510 worktree; its exact LFS object was resolved by the registered SHA256/OID from the shared cache, not regenerated",
        },
        "sources_sha256": {rel: sha(root / rel) for rel in RUNTIME_SOURCES},
        "exclusion_source_sha256": {rel: sha(root / rel) for rel in exclusion_source_paths(root)},
    }
    out.mkdir(parents=True, exist_ok=True)
    registration["runtime_assets"] = recover_runtime_assets(
        root, out / "runtime-artifacts" / "runtime", registration["runtime_identity_sha256"]
    )
    registration["runtime"]["exact_root_native_probe"] = registration["runtime_assets"]["exact_root_native_probe"]
    registration["runtime"]["exact_root_tablebase"] = registration["runtime_assets"]["exact_root_tablebase"]
    for name, component in components.items():
        source = root / component["artifact_source"]
        destination = out / "runtime-artifacts" / name
        shutil.copytree(source, destination)
        component["artifact"] = str(destination.relative_to(root))
        if {f: sha(destination / f) for f in component["files"]} != component["files"]:
            raise ValueError(f"copied_component_identity:{name}")
    (out / "openings.jsonl").write_bytes(suite)
    (out / "registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    freeze = {
        "status": "frozen_before_candidate_inference_or_games",
        "registration_sha256": sha(out / "registration.json"),
        "suite_sha256": sha(out / "openings.jsonl"),
    }
    (out / "execution-freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    preflight(root)
    return freeze


def command(
    root: Path, index: int, opening: dict[str, Any] | None = None
) -> list[str]:
    out = root / OUT
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    if opening is None:
        opening = read_rows(out / "openings.jsonl")[index]
    opening_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(opening) + b"\n"
    if opening_path.exists() and opening_path.read_bytes() != payload:
        raise ValueError(f"seed511_batch_opening_conflict:{index}")
    if not opening_path.exists():
        opening_path.write_bytes(payload)
    return resolved_command(root, index)


def resolved_command(root: Path, index: int) -> list[str]:
    out = root / OUT
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    artifacts = root / OUT / "runtime-artifacts"
    games_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    resolved_registration = json.loads((out / "registration.json").read_text())
    options = json.dumps(base.arena_search_overrides(resolved_registration), sort_keys=True)
    runtime_assets = json.loads((out / "registration.json").read_text())["runtime_assets"]
    return [
        sys.executable, "-m", "ml.alphazero_lite.arena",
        "--challenger", str(artifacts / "E4"), "--current", str(artifacts / "initializer"),
        "--challenger-policy-artifact", str(artifacts / "E4"),
        "--challenger-value-artifact", str(artifacts / "initializer"),
        "--games", "2", "--games-per-opening", "2", "--workers", "1",
        "--challenger-simulations", "128", "--current-simulations", "128", "--c-puct", "1.25",
        "--seed-contract", "azlite_eval_seed_v2", "--challenger-search-options-json", options,
        "--current-search-options-json", options, "--random-opening-plies", "0", "--max-moves", "200",
        "--exact-root-native-probe", str(root / runtime_assets["exact_root_native_probe"]["artifact"]),
        "--exact-root-tablebase", str(root / runtime_assets["exact_root_tablebase"]["artifact"]),
        "--seed", str(SEED + index), "--opening-prefixes-jsonl", str(opening_path),
        "--suite-sha256", sha(opening_path), "--game-jsonl", str(games_path), "--out", str(report_path),
    ]


def resume_plan(root: Path, openings: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[int]:
    """Validate committed pairs and preserve any outcome-bearing interrupted batch."""
    if len(ledger) % 2:
        raise ValueError("seed511_partial_ledger_pair")
    committed = {int(row["opening_index"]) for row in ledger}
    if len(ledger) != 2 * len(committed) or any(i not in range(len(openings)) for i in committed):
        raise ValueError("seed511_duplicate_or_invalid_ledger_game")
    from ml.alphazero_lite.verify_seed511_fixed_composition_screen import verify_pair

    for index in committed:
        verify_pair(root, index, openings[index], ledger)
    batch = root / OUT / "batches"
    pending = []
    for index in range(len(openings)):
        if index in committed:
            continue
        files = (
            batch / f"games-{index:04d}.jsonl",
            batch / f"report-{index:04d}.json",
            batch / f"opening-{index:04d}.jsonl",
        )
        if any(path.exists() for path in files):
            raise ValueError(f"seed511_interrupted_pair_preserved:{index}")
        pending.append(index)
    return pending


def run(root: Path = ROOT, *, execute: bool = False) -> dict[str, Any]:
    if not execute:
        raise ValueError("execution_requires_explicit_--execute")
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    freeze_path = out / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    preflight(root, require_unstarted=False)
    openings = read_rows(out / "openings.jsonl")
    if len(openings) != OPENINGS:
        raise ValueError("seed511_opening_count")
    ledger_path = out / "outcomes.jsonl"
    pending_path = ledger_path.with_suffix(".jsonl.pending")
    if pending_path.exists():
        raise ValueError("seed511_pending_ledger_requires_recovery")
    ledger = read_rows(ledger_path) if ledger_path.exists() else []
    pending_indices = resume_plan(root, openings, ledger)
    pointer = materialize_bound_tablebase(root, registration)
    try:
        return _execute_pending(
            root, out, registration, freeze, openings, ledger, pending_indices
        )
    finally:
        restore_bound_tablebase(root, registration, pointer)


def _execute_pending(
    root: Path,
    out: Path,
    registration: dict[str, Any],
    freeze: dict[str, Any],
    openings: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    pending_indices: list[int],
) -> dict[str, Any]:
    ledger_path = out / "outcomes.jsonl"
    pending_path = ledger_path.with_suffix(".jsonl.pending")
    for index in pending_indices:
        argv = command(root, index, openings[index])
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "opening_index": index,
                "argv": argv,
                "component_assignments": {
                    "challenger_policy": "E4",
                    "challenger_value": "initializer",
                    "challenger_policy_trunk": "E4 own trunk",
                    "challenger_value_trunk": "initializer own trunk",
                    "opponent": "complete initializer artifact",
                },
                "component_hashes": registration["components"],
                "runtime": registration["runtime"],
                "registration_sha256": freeze["registration_sha256"],
                "suite_sha256": freeze["suite_sha256"],
            }, sort_keys=True) + "\n")
        subprocess.run(argv, cwd=root, check=True)
        games_path = out / "batches" / f"games-{index:04d}.jsonl"
        report_path = out / "batches" / f"report-{index:04d}.json"
        report = json.loads(report_path.read_text())
        report["seed511_composition"] = {
            "policy_source": "E4", "value_source": "initializer",
            "policy_trunk": "E4 own trunk", "value_trunk": "initializer own trunk",
            "opponent": "complete initializer artifact",
            "components": registration["components"],
            "runtime_assets": registration["runtime_assets"],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        games = read_rows(games_path)
        if len(games) != 2 or {int(g["challenger_player"]) for g in games} != {0, 1}:
            raise ValueError(f"seed511_incomplete_pair:{index}")
        for position, game in enumerate(games):
            ledger.append({
                "opening_index": index,
                "challenger_player": int(game["challenger_player"]),
                "winner": game["winner"],
                "batch_game_position": position,
                "batch_game_sha256": sha(games_path),
                "batch_report_sha256": sha(report_path),
                "batch_opening_sha256": sha(out / "batches" / f"opening-{index:04d}.jsonl"),
            })
        pending_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ledger))
        pending_path.replace(ledger_path)
    if len(ledger) != GAMES:
        raise ValueError("seed511_final_game_count")
    result = analyze(ledger)
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def preflight(
    root: Path = ROOT, *, require_unstarted: bool = True
) -> dict[str, Any]:
    """Read-only registration, source, artifact, exclusion, and suite gate."""
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    freeze_path = out / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    suite_path = out / "openings.jsonl"
    if (
        registration.get("status") != "frozen_before_candidate_inference_or_games"
        or freeze.get("status") != "frozen_before_candidate_inference_or_games"
        or sha(registration_path) != freeze.get("registration_sha256")
        or sha(suite_path) != freeze.get("suite_sha256")
        or sha(suite_path) != registration["suite"]["sha256"]
    ):
        raise ValueError("seed511_freeze_mismatch")
    verify_frozen_inputs(root, registration)
    identities, accounting = reconstruct_exclusions(root)
    if accounting != registration["exclusions"]:
        raise ValueError("seed511_exclusion_accounting_mismatch")
    openings = read_rows(suite_path)
    regenerated, rejected = make_suite(identities)
    if (
        len(openings) != OPENINGS
        or suite_bytes(regenerated) != suite_path.read_bytes()
        or rejected != registration["suite"]["rejected"]
    ):
        raise ValueError("seed511_suite_generation_mismatch")
    if any(
        canonical_identity(row["state"]) in identities["canonical"]
        or float32_identity(row["state"]) in identities["float32"]
        for row in openings
    ):
        raise ValueError("seed511_excluded_opening_identity")
    if require_unstarted and (
        (out / "outcomes.jsonl").exists() or (out / "commands.jsonl").exists()
    ):
        raise ValueError("seed511_preflight_after_execution_started")
    return {
        "status": "preflight_passed_before_inference_or_games",
        "registration_sha256": freeze["registration_sha256"],
        "suite_sha256": freeze["suite_sha256"],
        "openings": len(openings),
        "excluded_canonical_identities": len(identities["canonical"]),
        "excluded_float32_identities": len(identities["float32"]),
    }


def analyze(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    points = {"challenger": 1.0, "draw": 0.5, "current": 0.0, "initializer": 0.0}
    by_opening = {i: [] for i in range(OPENINGS)}
    for row in ledger:
        by_opening[int(row["opening_index"])].append(points[row["winner"]])
    if any(len(scores) != 2 for scores in by_opening.values()):
        raise ValueError("seed511_analysis_pairing")
    cluster = np.asarray([np.mean(by_opening[i]) for i in range(OPENINGS)])
    boot = cluster[rng.integers(0, OPENINGS, (RESAMPLES, OPENINGS))].mean(axis=1)
    score = float(cluster.mean())
    lower = float(np.quantile(boot, 0.05))
    return {
        "games": len(ledger),
        "wins": sum(row["winner"] == "challenger" for row in ledger),
        "draws": sum(row["winner"] == "draw" for row in ledger),
        "losses": sum(row["winner"] in ("current", "initializer") for row in ledger),
        "score": score,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "pass": lower > .5,
        "promotion": False,
        "fresh_confirmation": False,
        "interpretation": "exploratory fixed-composition screen only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.execute:
        result = run(execute=True)
    else:
        parser.error("choose --prepare or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
