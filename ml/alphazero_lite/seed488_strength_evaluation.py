"""Preparation and resumable execution for the prospective seed488 arena.

Importing this module never runs an arena. Game execution requires the explicit
``--execute`` flag after review of the frozen protocol.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed488-frozen-strength-evaluation")
EXECUTION_FREEZE = "execution-freeze.json"
CORRECTED_ATTEMPT_NAMESPACE = "resume-corrected"
SEED = 488
OPENING_COUNT = 512
GAMES_PER_OPENING = 2
BOOTSTRAP_REPLICATES = 20_000
INITIALIZER = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
TRAJECTORY = Path("docs/data/seed479-population-ablation/trajectory-evidence.npz")
SEED479_RECEIPT = Path("docs/data/seed479-population-ablation/receipt.json")
SEED479_RECEIPT_SHA256 = (
    "f7c5edd6e6bedd496b145721edcd77196ccdb32e127340c81b1546a6db816314"
)
SEED479_REGISTRATION_SHA256 = (
    "4e2a03af089593f1ae32abc26f8a407bb5d980be7100776c091224a7ae0b2fb4"
)
SEED479_EVIDENCE_SHA256 = (
    "9056f7654b1dca824eccc783c04b7ad1a68140667e28068f665d304eea9ce2fe"
)
BASE_EXCLUSIONS = Path("docs/data/seed482-exclusion-manifest.json.gz")
RECONSTRUCTED_EXCLUSIONS = Path("docs/data/seed483-historical-probe-identities.json.gz")
EXTENDED_EXCLUSIONS = Path("docs/data/seed484-extended-probe-union.json.gz")
BASE_EXCLUSIONS_SHA256 = (
    "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d"
)
RECONSTRUCTED_EXCLUSIONS_SHA256 = (
    "cb11ea5912c530f4a6c01a913b9e5083c454f639d24669a4675ef6d1c2e45c9b"
)
EXTENDED_EXCLUSIONS_SHA256 = (
    "f419d615724f8f741fb4a58bb060901d4359e4f975ca64bfa184da4e892c10cd"
)
EXTENDED_UNION_PAIRS_SHA256 = (
    "d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6"
)
ARENA = Path("ml/alphazero_lite/arena.py")
RUNTIME_POLICY = Path("model-artifact/current/search_policy.json")
NATIVE_PROBE = Path("model-artifact/runtime/kalah_v1_tablebase")
TABLEBASE = Path("model-artifact/runtime/kalah_v1_21.kvtb")
RUNTIME_SOURCES = (
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
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/opening_exclusion_contract.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/opening_cache.py",
    "ml/alphazero_lite/report_validation.py",
    "ml/alphazero_lite/shadow_root_q.py",
)
INITIAL_STATE = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def state_pair(state: dict[str, Any]) -> tuple[str, str]:
    return canonical_identity(state), float32_identity(state)


def reconstructable_exclusions(root: Path) -> set[tuple[str, str]]:
    """Read only authenticated, already reconstructed seed482/483 exclusions."""
    root = root.resolve()
    base_path = root / BASE_EXCLUSIONS
    raw_base = base_path.read_bytes()
    if sha256_bytes(raw_base) != BASE_EXCLUSIONS_SHA256:
        raise ValueError("base_exclusion_file_hash")
    base = json.loads(gzip.decompress(raw_base))
    if (
        base.get("manifest_sha256")
        != "38e97a88041afd3a7f2e580f30155d17fae93f77ddf9267a6e663ca644b758b8"
    ):
        raise ValueError("base_exclusion_manifest_identity")
    add_path = root / RECONSTRUCTED_EXCLUSIONS
    raw_additions = add_path.read_bytes()
    if sha256_bytes(raw_additions) != RECONSTRUCTED_EXCLUSIONS_SHA256:
        raise ValueError("reconstructed_exclusion_file_hash")
    additions = json.loads(gzip.decompress(raw_additions))
    if (
        additions.get("union_sha256")
        != "74afb0d0a08d1c03d132124089213735843c1eb19497c3af653fdc672cbfc4bd"
    ):
        raise ValueError("reconstructed_exclusion_union_identity")
    pairs = {
        (str(item["canonical_identity"]), str(item["float32_identity"]))
        for item in base["identities"]
    }
    pairs.update(
        (str(item["canonical_identity"]), str(item["float32_identity"]))
        for item in additions["identities"]
    )
    extended_path = root / EXTENDED_EXCLUSIONS
    raw_extended = extended_path.read_bytes()
    if sha256_bytes(raw_extended) != EXTENDED_EXCLUSIONS_SHA256:
        raise ValueError("extended_exclusion_file_hash")
    extended = json.loads(gzip.decompress(raw_extended))
    pairs.update(
        (str(item["canonical_identity"]), str(item["float32_identity"]))
        for item in extended["seed467_additional_identities"]
    )
    pair_digest = sha256_bytes(canonical_json([list(pair) for pair in sorted(pairs)]))
    if (
        len(pairs) != 149_756
        or extended.get("final_union_pair_count") != 149_756
        or extended.get("final_union_pairs_sha256") != EXTENDED_UNION_PAIRS_SHA256
        or pair_digest != EXTENDED_UNION_PAIRS_SHA256
        or extended.get("status") != "incomplete_seed461_replay_gap"
    ):
        raise ValueError("extended_exclusion_union_identity")
    return pairs


def make_suite(root: Path, *, seed: int = SEED) -> list[dict[str, Any]]:
    """Generate a deterministic, exclusion-screened set of opening states.

    This uses rules-only random prefixes; it never calls a model, search, or
    arena. Every selected opening root is screened against the reconstructable
    identity union and earlier selected opening roots.
    """
    excluded = reconstructable_exclusions(root)
    excluded_canonical = {pair[0] for pair in excluded}
    excluded_float32 = {pair[1] for pair in excluded}
    selected_canonical: set[str] = set()
    selected_float32: set[str] = set()
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    attempts = 0
    while len(entries) < OPENING_COUNT and attempts < 1_000_000:
        attempts += 1
        game = KalahGame.from_state(INITIAL_STATE)
        absolute: list[int] = []
        collision = False
        target_plies = rng.randint(4, 12)
        for _ in range(target_plies):
            if game.over():
                collision = True
                break
            legal = game.possible_moves()
            if not legal:
                collision = True
                break
            relative = rng.choice(legal)
            absolute_move = game.pit_index(relative)
            absolute.append(absolute_move)
            if not game.move(absolute_move):
                raise ValueError("generated_opening_move_invalid")
        if collision or game.over():
            continue
        state = game.to_state()
        pair = state_pair(state)
        if (
            pair[0] in excluded_canonical
            or pair[1] in excluded_float32
            or pair[0] in selected_canonical
            or pair[1] in selected_float32
        ):
            continue
        selected_canonical.add(pair[0])
        selected_float32.add(pair[1])
        relative_prefix: list[int] = []
        replay = KalahGame.from_state(INITIAL_STATE)
        for move in absolute:
            relative_prefix.append(move - replay.current_player * 6)
            if not replay.move(move):
                raise ValueError("generated_opening_replay_invalid")
        if replay.to_state() != state:
            raise ValueError("generated_opening_state_mismatch")
        entries.append(
            {
                "opening_index": len(entries),
                "opening_contract": "arena_player_relative_v2",
                "prefix_moves": relative_prefix,
                "source_prefix_moves_absolute": absolute,
                "state": state,
                "state_hash": hashlib.sha256(canonical_json(state)).hexdigest(),
            }
        )
    if len(entries) != OPENING_COUNT:
        raise ValueError("opening_suite_eligibility_shortfall")
    return entries


def suite_bytes(entries: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(entry) + b"\n" for entry in entries)


def batch_opening_identity(
    registration: dict[str, Any], opening: dict[str, Any], batch_path: Path
) -> dict[str, Any]:
    """Bind the local single-row arena batch to its parent-suite entry."""
    parent_index = int(opening["opening_index"])
    if not 0 <= parent_index < OPENING_COUNT:
        raise ValueError("batch_parent_opening_index")
    expected_seed = SEED + parent_index
    parent_digest = registration["opening_suite"]["opening_suite_sha256"]
    batch_digest = sha256_file(batch_path)
    if batch_path.read_bytes() != canonical_json(opening) + b"\n":
        raise ValueError("batch_parent_opening_association")
    return {
        "parent_suite_sha256": parent_digest,
        "batch_suite_sha256": batch_digest,
        "parent_opening_index": parent_index,
        "parent_entry_state_hash": opening["state_hash"],
        "local_opening_index": 0,
        "base_seed": expected_seed,
        "seed_derivation": "488 + parent_opening_index",
    }


def arena_search_overrides(registration: dict[str, Any]) -> dict[str, Any]:
    """Return per-side search settings, excluding arena-level opening options."""
    options = dict(registration["runtime"]["search_options"])
    options.pop("random_opening_plies", None)
    return options


def arena_effective_search_options(registration: dict[str, Any]) -> dict[str, Any]:
    """Resolve JSON overrides to the search options reported by arena."""
    options = arena_search_overrides(registration)
    options.pop("root_prior_transform", None)
    return options


def build_arena_command(
    registration: dict[str, Any],
    artifacts: Path,
    opening_path: Path,
    report_path: Path,
    games_path: Path,
    batch_identity: dict[str, Any],
) -> list[str]:
    """Construct the exact frozen arena invocation for one paired opening."""
    overrides = json.dumps(arena_search_overrides(registration), sort_keys=True)
    return [
        sys.executable,
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(artifacts / "F"),
        "--current",
        str(artifacts / "initializer"),
        "--games",
        "2",
        "--games-per-opening",
        "2",
        "--workers",
        "1",
        "--challenger-simulations",
        "1200",
        "--current-simulations",
        "1200",
        "--c-puct",
        "1.25",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--challenger-search-options-json",
        overrides,
        "--current-search-options-json",
        overrides,
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--seed",
        str(batch_identity["base_seed"]),
        "--opening-prefixes-jsonl",
        str(opening_path),
        "--suite-sha256",
        batch_identity["batch_suite_sha256"],
        "--game-jsonl",
        str(games_path),
        "--out",
        str(report_path),
    ]


def prepare(root: Path) -> dict[str, Any]:
    """Freeze the deterministic suite and protocol without exporting or evaluating."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    if (out / "failure-provenance.json").exists() or (out / EXECUTION_FREEZE).exists():
        raise ValueError("cannot_prepare_after_seed488_execution_attempt")
    if any((out / name).exists() for name in ("outcomes.jsonl", "results.json")):
        raise ValueError("cannot_supersede_seed488_after_outcomes_exist")
    suite = suite_bytes(make_suite(root))
    suite_path = out / "openings.jsonl"
    seed479_receipt = json.loads((root / SEED479_RECEIPT).read_text())
    if sha256_file(root / SEED479_RECEIPT) != SEED479_RECEIPT_SHA256:
        raise ValueError("seed479_receipt_file_hash")
    if seed479_receipt.get("status") != "seed479_population_ablation_bound":
        raise ValueError("seed479_receipt_status")
    if (
        sha256_file(root / "docs/data/seed479-population-ablation/registration.json")
        != SEED479_REGISTRATION_SHA256
    ):
        raise ValueError("seed479_registration_hash")
    if (
        sha256_file(root / "docs/data/seed479-population-ablation/evidence.json")
        != SEED479_EVIDENCE_SHA256
    ):
        raise ValueError("seed479_evidence_hash")
    if sha256_file(root / TRAJECTORY) != seed479_receipt.get("trajectory-evidence.npz"):
        raise ValueError("seed479_trajectory_receipt_mismatch")
    runtime_files = (
        *map(Path, RUNTIME_SOURCES),
        RUNTIME_POLICY,
        NATIVE_PROBE,
        TABLEBASE,
    )
    runtime_identity = {str(path): sha256_file(root / path) for path in runtime_files}
    registration = {
        "schema": "seed488-frozen-strength-evaluation-v2",
        "supersedes": "PR #488 preparation registration with parent-suite digest passed to single-opening arena batches; this is a preparation-only correction before any outcomes",
        "status": "frozen_before_execution",
        "interpretation": "exploratory direct strength evaluation on a newly registered suite; not fresh confirmation or an independently untouched holdout",
        "fresh_confirmation_authorized": False,
        "seed479_and_480_classification": "promising exploratory results; preserved as exploratory",
        "arms": {
            "seed479_receipts": {
                "receipt_sha256": SEED479_RECEIPT_SHA256,
                "registration_sha256": SEED479_REGISTRATION_SHA256,
                "evidence_sha256": SEED479_EVIDENCE_SHA256,
            },
            "F": {
                "checkpoint_source": str(TRAJECTORY),
                "trajectory_sha256": sha256_file(root / TRAJECTORY),
                "parameter_state_sha256": "29f9a687dcdd8f39ca09bb7cf8fe6ba7ea268f4fdabc6b2af04f4d93a06402f3",
                "state_keys": "F_step_16_post_parameter_00 through _21 in stored model parameter order",
            },
            "initializer": {
                "checkpoint_source": str(INITIALIZER),
                "checkpoint_sha256": sha256_file(root / INITIALIZER),
            },
        },
        "export_semantics": "both checkpoints exported using the same residual_v3 (96,3), kalah_v3 runtime exporter; shared runtime search_policy sidecar; synthetic policy/value parity against frozen parameter states is required before any real registration/execution",
        "runtime": {
            "identities_sha256": runtime_identity,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "native_runtime": "same compiled native exact-root probe and tablebase for both arms, bound by hashes",
            "simulations_per_move": 1200,
            "c_puct": 1.25,
            "workers": 1,
            "seed_contract": "azlite_eval_seed_v2; batch-local opening index is 0; base seed = 488 + parent opening index; arena deterministically binds the single batch opening state, seat, game-within-opening, and ply",
            "search_options": {
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
                "root_prior_transform": None,
                "random_opening_plies": 0,
            },
            "max_moves": 200,
            "max_moves_policy": "At 200 decision plies, stop without sweeping remaining pits; compare captured stores at the cap. A game ending earlier uses Kalah terminal sweeping. Recorded trajectory length must be terminal before 200 or exactly 200.",
            "root_prior_transform": None,
            "policy_and_value_transforms": None,
            "exact_root": "shared frozen sidecar policy; threshold 16, final_score_margin objective, highest legal network prior then lowest move index tie rule, exact leaf solve disabled",
            "seat_pairing": "each opening is played twice with F in player 0 and player 1; same opening prefix and other settings",
        },
        "opening_suite": {
            "generator": "Python random.Random(488), legal rules-only prefixes of 4..12 plies; reject opening-root identity collisions against independently reconstructed seed482/483/484 identity pairs and previous selected roots",
            "generator_seed": SEED,
            "count": OPENING_COUNT,
            "games_per_opening": GAMES_PER_OPENING,
            "game_count": OPENING_COUNT * GAMES_PER_OPENING,
            "opening_suite_sha256": sha256_bytes(suite),
            "batch_identity_contract": {
                "parent_suite_sha256": sha256_bytes(suite),
                "batch_suite_sha256": "SHA-256 of the exact one-entry JSONL passed to arena --opening-prefixes-jsonl",
                "parent_opening_index": "opening_index in openings.jsonl; arena-local index is 0",
                "parent_entry_identity": "state_hash of the parent suite entry, validated against exact batch bytes",
                "seed_derivation": "registered base seed = 488 + parent_opening_index; batch-local index does not affect seed",
            },
            "eligible_against_reconstructable_exclusions": True,
            "exclusions": {
                "seed482_base_manifest_sha256": BASE_EXCLUSIONS_SHA256,
                "seed483_reconstructed_union_sha256": "74afb0d0a08d1c03d132124089213735843c1eb19497c3af653fdc672cbfc4bd",
                "seed483_reconstructed_union_file_sha256": RECONSTRUCTED_EXCLUSIONS_SHA256,
                "seed484_extended_union_file_sha256": EXTENDED_EXCLUSIONS_SHA256,
                "seed484_extended_union_sha256": EXTENDED_UNION_PAIRS_SHA256,
                "seed484_extended_union_pair_count": 149_756,
                "limitation": "non-overlap with unresolved seed461 continuation population cannot be certified; seed461 root openings are represented where reconstructable, but trajectory state identities are not",
            },
        },
        "implementation_sha256": sha256_file(
            root / Path("ml/alphazero_lite/seed488_strength_evaluation.py")
        ),
        "generator_dependency_sha256": {
            "ml/alphazero_lite/seed482_fresh_confirmation.py": sha256_file(
                root / "ml/alphazero_lite/seed482_fresh_confirmation.py"
            )
        },
        "independent_verifier_sha256": sha256_file(
            root / Path("ml/alphazero_lite/verify_seed488_strength_evaluation.py")
        ),
        "synthetic_tests_sha256": sha256_file(
            root / Path("ml/alphazero_lite/test_seed488_strength_evaluation.py")
        ),
        "generator_runtime": {"python_version": sys.version, "platform": sys.platform},
        "primary_score": "F wins + 0.5 * draws, divided by all 1,024 games",
        "bootstrap": {
            "unit": "512 opening clusters; resample with replacement and carry both seat games together",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "advance_rule": "advance only if one-sided 95% percentile lower score bound > 0.50",
            "two_sided_interval": "95% percentile interval also reported descriptively",
        },
        "promotion": "No promotion from this experiment alone.",
        "resumption": "one atomic paired opening batch at a time; completed opening batches are receipt-bound and skipped on resume; duplicate, malformed, or incomplete batches reject",
        "trajectory_contract": "arena.py at the runtime source hash; opening prefixes and trajectories are player-relative v2 moves converted by KalahGame.pit_index at each ply; rules use KalahGame.move including extra turns and terminal sweep; stored margin is challenger captured store minus current captured store; winner is the sign of that margin",
        "estimated_execution_cost": "1,024 games × at most 200 decision plies × 1,200 simulations = 245,760,000 simulation budget upper bound, plus exact-root probes; actual wall time depends on native runtime throughput and game lengths. No arena run is performed in preparation.",
    }
    previous_files = {}
    for filename in ("registration.json", "openings.jsonl", "receipt.json"):
        original = out / filename
        if original.is_file():
            preserved = (
                out
                / f"{filename.removesuffix('.json').removesuffix('.jsonl')}-pre-batch-identity{original.suffix}"
            )
            if not preserved.exists():
                shutil.copy2(original, preserved)
            previous_files[f"pre_batch_identity:{filename}"] = {
                "preserved_as": preserved.name,
                "sha256": sha256_file(preserved),
                "mtime_ns": preserved.stat().st_mtime_ns,
            }
            preserved = (
                out
                / f"{filename.removesuffix('.json').removesuffix('.jsonl')}-pr487{original.suffix}"
            )
            if not preserved.exists():
                shutil.copy2(original, preserved)
            previous_files[filename] = {
                "preserved_as": preserved.name,
                "sha256": sha256_file(preserved),
                "mtime_ns": preserved.stat().st_mtime_ns,
            }
    prior_supersession = out / "supersession-pr489.json"
    if not prior_supersession.exists() and (out / "supersession.json").is_file():
        shutil.copy2(out / "supersession.json", prior_supersession)
    if prior_supersession.is_file():
        previous_files["supersession:pr489"] = {
            "preserved_as": prior_supersession.name,
            "sha256": sha256_file(prior_supersession),
            "mtime_ns": prior_supersession.stat().st_mtime_ns,
        }
    supersession = {
        "status": "seed488_batch_identity_preparation_superseded_before_execution",
        "supersedes_pr": 488,
        "preserved_originals": previous_files,
        "no_outcomes_existed_at_supersession": True,
        "scope_unchanged": {
            "games": 1024,
            "paired_opening_bootstrap": True,
            "exploratory": True,
            "promotion": False,
        },
    }
    supersession_path = out / "supersession.json"
    supersession_path.write_text(
        json.dumps(supersession, indent=2, sort_keys=True) + "\n"
    )
    suite_path.write_bytes(suite)
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    publish_receipt(
        out,
        ("registration.json", "openings.jsonl", "supersession.json"),
        status="seed488_prepared_not_executed",
    )
    return registration


def score_winner(winner: str) -> float:
    scores = {"F": 1.0, "draw": 0.5, "initializer": 0.0}
    try:
        return scores[winner]
    except KeyError as error:
        raise ValueError("unknown_game_outcome") from error


def normalized_winner(row: dict[str, Any]) -> str:
    winner = row.get("winner")
    if winner in {"challenger", "current"}:
        return "F" if winner == "challenger" else "initializer"
    if winner in {"F", "initializer", "draw"}:
        return str(winner)
    raise ValueError("unknown_game_outcome")


def completed_opening_indices(rows: list[dict[str, Any]]) -> set[int]:
    seats: dict[int, set[int]] = {}
    for row in rows:
        opening = int(row["opening_index"])
        seat = int(row["F_player"])
        if not 0 <= opening < OPENING_COUNT or seat not in (0, 1):
            raise ValueError("duplicate_or_invalid_game_identity")
        observed = seats.setdefault(opening, set())
        if seat in observed:
            raise ValueError("duplicate_or_invalid_game_identity")
        normalized_winner(row)
        observed.add(seat)
    if any(value != {0, 1} for value in seats.values()):
        raise ValueError("partial_or_incomplete_opening_batch")
    return set(seats)


def paired_scores(rows: list[dict[str, Any]]) -> tuple[float, list[float]]:
    expected = {(opening, seat) for opening in range(OPENING_COUNT) for seat in (0, 1)}
    seen: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["opening_index"]), int(row["F_player"]))
        if key not in expected or key in seen:
            raise ValueError("duplicate_or_invalid_game_identity")
        seen[key] = score_winner(normalized_winner(row))
    if seen.keys() != expected:
        raise ValueError("missing_games")
    opening_scores = [
        float((seen[(i, 0)] + seen[(i, 1)]) / 2) for i in range(OPENING_COUNT)
    ]
    return float(np.mean(opening_scores)), opening_scores


def opening_bootstrap(opening_scores: list[float]) -> dict[str, Any]:
    if len(opening_scores) != OPENING_COUNT:
        raise ValueError("bootstrap_opening_count")
    values = np.asarray(opening_scores, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    scores = values[samples].mean(axis=1)
    return {
        "method": "opening-paired percentile bootstrap; resample 512 opening clusters, carrying both seats together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "confidence_level": 0.95,
        "one_sided_lower": float(np.quantile(scores, 0.05)),
        "two_sided_interval": [
            float(np.quantile(scores, 0.025)),
            float(np.quantile(scores, 0.975)),
        ],
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def run(root: Path, *, execute: bool = False) -> None:
    """Resume one two-seat arena batch per opening after explicit opt-in."""
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    suite_path = out / "openings.jsonl"
    if not registration_path.is_file() or not suite_path.is_file():
        raise ValueError("seed488_protocol_not_prepared")
    registration = json.loads(registration_path.read_text())
    if sha256_file(suite_path) != registration["opening_suite"]["opening_suite_sha256"]:
        raise ValueError("opening_suite_receipt_mismatch")
    if not execute:
        raise ValueError("execution_requires_explicit_--execute_after_protocol_review")
    outcome_path_pending = out / "outcomes.jsonl.pending"
    if outcome_path_pending.exists():
        raise ValueError("interrupted_pending_commit_requires_operator_recovery")
    if not (out / "opening-0000-recovery-receipt.json").is_file():
        raise ValueError("seed488_opening_zero_recovery_receipt_required")
    from ml.alphazero_lite.seed488_replay_recovery import verify_receipt

    recovery = verify_receipt(root)
    ledger_path = out / "outcomes.jsonl"
    if not ledger_path.is_file():
        raise ValueError("seed488_opening_zero_recovery_import_required")
    opening_zero_rows = [
        row for row in _read_rows(ledger_path) if int(row.get("opening_index", -1)) == 0
    ]
    if len(opening_zero_rows) != 2 or any(
        row.get("recovery_receipt_sha256")
        != sha256_file(out / "opening-0000-recovery-receipt.json")
        for row in opening_zero_rows
    ):
        raise ValueError("seed488_opening_zero_recovery_provenance")
    if recovery.get("status") != "validated_for_recovery_import":
        raise ValueError("seed488_recovery_receipt_status")
    if registration.get("fresh_confirmation_authorized") is not False:
        raise ValueError("fresh_confirmation_gate_must_remain_closed")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(root),
        ],
        cwd=root,
        check=True,
    )
    opening_rows = _read_rows(suite_path)
    outcome_path = ledger_path
    completed_rows = _read_rows(outcome_path) if outcome_path.exists() else []
    completed_openings = completed_opening_indices(completed_rows)
    opening_by_index = {int(row["opening_index"]): row for row in opening_rows}
    for index in sorted(completed_openings):
        _validate_committed_batch(
            out, opening_by_index[index], completed_rows, registration
        )
    execution_freeze_path = out / EXECUTION_FREEZE
    if not execution_freeze_path.is_file():
        raise ValueError("seed488_execution_freeze_missing")
    execution_freeze = json.loads(execution_freeze_path.read_text())
    if execution_freeze.get("status") != "frozen_before_execution_review_required":
        raise ValueError("seed488_execution_freeze_status")
    if execution_freeze.get("execution_namespace") != CORRECTED_ATTEMPT_NAMESPACE:
        raise ValueError("seed488_execution_namespace")
    artifacts = out / "runtime-artifacts"
    artifacts.mkdir(exist_ok=True)
    for arm in ("F", "initializer"):
        artifact = artifacts / arm
        if not artifact.exists():
            checkpoint = _materialize_endpoint(root, arm, out)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml.alphazero_lite.export_artifact",
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(artifact),
                    "--version",
                    f"seed488-{arm}",
                    "--model-type",
                    "residual_v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=root,
                check=True,
            )
            (artifact / "search_policy.json").write_bytes(
                (root / RUNTIME_POLICY).read_bytes()
            )
    artifact_manifest: dict[str, dict[str, str]] = {}
    for arm in ("F", "initializer"):
        artifact = artifacts / arm
        artifact_manifest[arm] = {
            name: sha256_file(artifact / name)
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
    manifest_path = out / "runtime-artifacts.json"
    serialized_manifest = json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != serialized_manifest:
            raise ValueError("resumed_runtime_artifact_manifest_mismatch")
    else:
        manifest_path.write_text(serialized_manifest)
    _validate_exported_artifacts(root, out, artifact_manifest)
    for entry in opening_rows:
        index = int(entry["opening_index"])
        if index in completed_openings:
            continue
        batch_out = out / CORRECTED_ATTEMPT_NAMESPACE
        one_opening = batch_out / f"opening-{index:04d}.jsonl"
        one_opening.parent.mkdir(parents=True, exist_ok=True)
        expected_opening_bytes = canonical_json(entry) + b"\n"
        if one_opening.exists():
            if one_opening.read_bytes() != expected_opening_bytes:
                raise ValueError(f"corrected_batch_opening_mismatch:{index}")
        else:
            one_opening.write_bytes(expected_opening_bytes)
        batch_identity = batch_opening_identity(registration, entry, one_opening)
        report = batch_out / f"report-{index:04d}.json"
        games = batch_out / f"games-{index:04d}.jsonl"
        if games.exists() or report.exists():
            raise ValueError(f"interrupted_uncommitted_batch:{index}")
        command = build_arena_command(
            registration,
            artifacts,
            one_opening,
            report,
            games,
            batch_identity,
        )
        runtime_contract = _batch_runtime_contract(
            registration,
            index,
            sha256_file(manifest_path),
            batch_identity,
        )
        subprocess.run(command, cwd=root, check=True)
        raw_games = _read_rows(games)
        if len(raw_games) != GAMES_PER_OPENING:
            raise ValueError("arena_pair_game_count")
        normalized = []
        for game_position, game in enumerate(raw_games):
            cp = int(game["challenger_player"])
            normalized.append(
                {
                    **game,
                    "opening_index": index,
                    "opening_state_hash": entry["state_hash"],
                    "F_player": cp,
                    "batch_game_position": game_position,
                    "batch_artifact_dir": CORRECTED_ATTEMPT_NAMESPACE,
                    "batch_game_log_sha256": sha256_file(games),
                    "batch_report_sha256": sha256_file(report),
                    "batch_opening_sha256": sha256_file(one_opening),
                    "batch_runtime_contract": runtime_contract,
                }
            )
        _validate_committed_batch(
            out, entry, [*completed_rows, *normalized], registration
        )
        append_rows_atomic(outcome_path, [*completed_rows, *normalized])
        completed_rows.extend(normalized)
        completed_openings.add(index)
    if len(completed_rows) != OPENING_COUNT * GAMES_PER_OPENING:
        raise ValueError("arena_outcome_count")
    score, by_opening = paired_scores(completed_rows)
    result = {
        "schema": "seed488-strength-evaluation-results-v1",
        "game_count": len(completed_rows),
        "score": score,
        "bootstrap": opening_bootstrap(by_opening),
        "advance": opening_bootstrap(by_opening)["one_sided_lower"] > 0.5,
        "promotion": False,
    }
    result_path = out / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    publish_receipt(
        out,
        (
            "registration.json",
            "openings.jsonl",
            "supersession.json",
            "failure-provenance.json",
            EXECUTION_FREEZE,
            "runtime-artifacts.json",
            "outcomes.jsonl",
            "results.json",
        ),
        receipt_name="execution-receipt.json",
    )


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if pending.exists() and pending.read_text(encoding="utf-8") != payload:
        raise ValueError("conflicting_pending_outcome_ledger")
    pending.write_text(payload, encoding="utf-8")
    pending.replace(path)


def _validate_committed_batch(
    out: Path,
    opening: dict[str, Any],
    rows: list[dict[str, Any]],
    registration: dict[str, Any],
) -> None:
    """Revalidate a committed pair before resume skips its arena invocation."""
    from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

    index = int(opening["opening_index"])
    batch_dirs = {
        row.get("batch_artifact_dir")
        for row in rows
        if int(row["opening_index"]) == index
    }
    if batch_dirs != {CORRECTED_ATTEMPT_NAMESPACE}:
        raise ValueError(f"committed_batch_artifact_namespace:{index}")
    batch_out = out / CORRECTED_ATTEMPT_NAMESPACE
    opening_path = batch_out / f"opening-{index:04d}.jsonl"
    batch_identity = batch_opening_identity(registration, opening, opening_path)
    expected_runtime_contract = _batch_runtime_contract(
        registration,
        index,
        sha256_file(out / "runtime-artifacts.json"),
        batch_identity,
    )
    pair = [row for row in rows if int(row["opening_index"]) == index]
    if len(pair) != 2 or {int(row["F_player"]) for row in pair} != {0, 1}:
        raise ValueError(f"committed_pair_coverage:{index}")
    games_path = batch_out / f"games-{index:04d}.jsonl"
    report_path = batch_out / f"report-{index:04d}.json"
    expected_opening = canonical_json(opening) + b"\n"
    game_rows = _read_rows(games_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if opening_path.read_bytes() != expected_opening or len(game_rows) != 2:
        raise ValueError(f"committed_batch_artifacts:{index}")
    if {int(item.get("challenger_player", -1)) for item in game_rows} != {0, 1}:
        raise ValueError(f"committed_batch_seats:{index}")
    for outcome in pair:
        position = int(outcome["batch_game_position"])
        game = game_rows[position]
        if (
            sha256_file(games_path) != outcome["batch_game_log_sha256"]
            or sha256_file(report_path) != outcome["batch_report_sha256"]
            or sha256_file(opening_path) != outcome["batch_opening_sha256"]
            or outcome["opening_state_hash"] != opening["state_hash"]
            or int(game["challenger_player"]) != int(outcome["F_player"])
            or outcome.get("batch_runtime_contract") != expected_runtime_contract
        ):
            raise ValueError(f"committed_batch_identity:{index}")
        _replay_game(opening, game, int(outcome["F_player"]))
        winner = game["winner"]
        expected_winner = {"challenger": "F", "current": "initializer"}.get(
            winner, winner
        )
        if normalized_winner(outcome) != expected_winner:
            raise ValueError(f"committed_batch_outcome:{index}")
    notes = report.get("notes", {})
    winners = [str(item.get("winner")) for item in game_rows]
    if (
        report.get("games_played") != 2
        or report.get("wins") != winners.count("challenger")
        or report.get("losses") != winners.count("current")
        or report.get("draws") != winners.count("draw")
        or notes.get("challenger_simulations") != 1200
        or notes.get("current_simulations") != 1200
        or notes.get("workers_used") != 1
        or notes.get("random_opening_plies") != 0
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("base_seed") != SEED + index
        or notes.get("suite_sha256") != expected_runtime_contract["batch_suite_sha256"]
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("exact_root_solve_threshold") != 16
        or notes.get("exact_root_native_probe_sha256")
        != registration["runtime"]["identities_sha256"][
            "model-artifact/runtime/kalah_v1_tablebase"
        ]
        or notes.get("exact_root_tablebase_sha256")
        != registration["runtime"]["identities_sha256"][
            "model-artifact/runtime/kalah_v1_21.kvtb"
        ]
        or notes.get("runtime_search_policy_sha256")
        != registration["runtime"]["identities_sha256"][
            "model-artifact/current/search_policy.json"
        ]
        or notes.get("runtime_search_policy_mode") != "puct_exact_root_hybrid"
        or notes.get("exact_root_objective") != "final_score_margin"
        or notes.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("exact_leaf_solve_mode") != "disabled"
        or notes.get("seed") != SEED + index
        or notes.get("search_options") != arena_effective_search_options(registration)
    ):
        raise ValueError(f"committed_batch_runtime:{index}")


def _batch_runtime_contract(
    registration: dict[str, Any],
    opening_index: int,
    artifact_manifest_sha256: str,
    batch_identity: dict[str, Any],
) -> dict[str, Any]:
    identities = registration["runtime"]["identities_sha256"]
    return {
        "games": 2,
        "games_per_opening": 2,
        "workers": 1,
        "challenger_simulations": 1200,
        "current_simulations": 1200,
        "c_puct": 1.25,
        "seed_contract": "azlite_eval_seed_v2",
        "base_seed": SEED + opening_index,
        "parent_suite_sha256": registration["opening_suite"]["opening_suite_sha256"],
        "batch_suite_sha256": batch_identity["batch_suite_sha256"],
        "parent_opening_index": opening_index,
        "parent_entry_state_hash": batch_identity["parent_entry_state_hash"],
        "local_opening_index": 0,
        "seed_derivation": "488 + parent_opening_index",
        "max_moves": 200,
        "opening_configuration": {"random_opening_plies": 0},
        "challenger_search_options": arena_search_overrides(registration),
        "current_search_options": arena_search_overrides(registration),
        "root_prior_transform": None,
        "challenger_root_prior_transform": None,
        "current_root_prior_transform": None,
        "challenger_value_transform": None,
        "current_value_transform": None,
        "opening_suite_sha256": registration["opening_suite"]["opening_suite_sha256"],
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "native_probe_sha256": identities["model-artifact/runtime/kalah_v1_tablebase"],
        "tablebase_sha256": identities["model-artifact/runtime/kalah_v1_21.kvtb"],
        "runtime_search_policy_sha256": identities[
            "model-artifact/current/search_policy.json"
        ],
        "exact_root_solve_threshold": 16,
        "batch_artifact_dir": CORRECTED_ATTEMPT_NAMESPACE,
    }


def publish_receipt(
    out: Path,
    names: tuple[str, ...],
    *,
    status: str = "seed488_strength_evaluation_complete",
    receipt_name: str = "receipt.json",
) -> None:
    receipt = {name: sha256_file(out / name) for name in names}
    receipt["status"] = status
    (out / receipt_name).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def _materialize_endpoint(root: Path, arm: str, out: Path) -> Path:
    import torch
    from ml.alphazero_lite import train

    if arm == "initializer":
        return root / INITIALIZER
    with np.load(root / TRAJECTORY, allow_pickle=False) as archive:
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        parameters = list(model.parameters())
        for index, parameter in enumerate(parameters):
            value = archive[f"F_step_16_post_parameter_{index:02d}"]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError("f_endpoint_parameter_shape")
            parameter.data.copy_(torch.from_numpy(value.copy()))
        flattened = np.concatenate(
            [p.detach().numpy().reshape(-1).astype("<f4") for p in parameters]
        )
        if (
            sha256_bytes(flattened.tobytes())
            != "29f9a687dcdd8f39ca09bb7cf8fe6ba7ea268f4fdabc6b2af04f4d93a06402f3"
        ):
            raise ValueError("f_endpoint_parameter_identity")
    checkpoint = train.checkpoint_from_model(model)
    path = out / "F-endpoint.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(checkpoint) or any(
                not np.array_equal(saved[key], checkpoint[key]) for key in checkpoint
            ):
                raise ValueError("preserved_f_endpoint_checkpoint_mismatch")
    else:
        np.savez(path, **checkpoint)
    return path


def _validate_exported_artifacts(
    root: Path, out: Path, manifest: dict[str, dict[str, str]]
) -> None:
    """Require the staged runtime exports to encode each frozen parameter state."""
    for arm in ("F", "initializer"):
        artifact = out / "runtime-artifacts" / arm
        checkpoint = (
            _materialize_endpoint(root, "F", out) if arm == "F" else root / INITIALIZER
        )
        with (
            np.load(checkpoint, allow_pickle=False) as expected,
            np.load(artifact / "model.npz", allow_pickle=False) as actual,
        ):
            if set(expected.files) != set(actual.files):
                raise ValueError(f"exported_checkpoint_keys:{arm}")
            for key in expected.files:
                if not np.array_equal(expected[key], actual[key]):
                    raise ValueError(f"exported_checkpoint_value:{arm}:{key}")
                exported_json = json.loads((artifact / "weights.json").read_text())
                if set(exported_json) != set(expected.files):
                    raise ValueError(f"exported_weights_json_keys:{arm}")
                if not np.array_equal(
                    np.asarray(exported_json[key], dtype=np.float32), expected[key]
                ):
                    raise ValueError(f"exported_weights_json_value:{arm}:{key}")
        if (artifact / "search_policy.json").read_bytes() != (
            root / RUNTIME_POLICY
        ).read_bytes():
            raise ValueError(f"exported_runtime_policy:{arm}")
        metadata = json.loads((artifact / "metadata.json").read_text())
        artifacts = metadata.get("artifacts", {})
        if (
            metadata.get("schema_version") != "azlite_model_v1"
            or metadata.get("version") != f"seed488-{arm}"
            or metadata.get("game") != "kalah"
            or metadata.get("rules_version") != "kalah_v1"
            or metadata.get("input_encoding") != "kalah_v3"
            or metadata.get("feature_count") != 27
            or metadata.get("policy_size") != 6
            or metadata.get("architecture", {}).get("model_type") != "residual_v3"
            or artifacts.get("weights_file") != "weights.json"
            or artifacts.get("weights_fallback_file") != "model.npz"
            or artifacts.get("weights_sha256") != sha256_file(artifact / "model.npz")
            or artifacts.get("weights_json_sha256")
            != sha256_file(artifact / "weights.json")
        ):
            raise ValueError(f"exported_metadata:{arm}")
        for name, digest in manifest[arm].items():
            if sha256_file(artifact / name) != digest:
                raise ValueError(f"exported_artifact_receipt:{arm}:{name}")


def verify(root: Path, results_path: Path | None = None) -> dict[str, Any]:
    """Read-only verifier for a fully receipt-bound outcome ledger."""
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    for name, expected_hash in receipt.items():
        if name == "status":
            continue
        if sha256_file(out / name) != expected_hash:
            raise ValueError(f"receipt_mismatch:{name}")
    if (
        reg.get("status") != "frozen_before_execution"
        or reg.get("fresh_confirmation_authorized") is not False
    ):
        raise ValueError("protocol_status")
    if reg.get("opening_suite", {}).get("opening_suite_sha256") != sha256_file(
        out / "openings.jsonl"
    ):
        raise ValueError("suite_binding")
    outcomes = results_path or out / "outcomes.jsonl"
    rows = _read_rows(outcomes)
    score, by_opening = paired_scores(rows)
    bootstrap = opening_bootstrap(by_opening)
    result = {
        "game_count": len(rows),
        "score": score,
        "bootstrap": bootstrap,
        "advance": bootstrap["one_sided_lower"] > 0.5,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicit opt-in; not run by preparation or verification",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Freeze suite/protocol only; does not run games",
    )
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.root), indent=2, sort_keys=True))
    elif args.verify:
        print(json.dumps(verify(args.root), indent=2, sort_keys=True))
    else:
        run(args.root, execute=args.execute)


if __name__ == "__main__":
    main()
