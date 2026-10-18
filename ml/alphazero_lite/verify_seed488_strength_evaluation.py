"""Independent read-only verification for seed488 strength outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed488-frozen-strength-evaluation")
EXECUTION_FREEZE = "execution-freeze.json"
OPENINGS = 512
REPLICATES = 20_000
SEED = 488
EXTENDED_EXCLUSION_SHA256 = (
    "f419d615724f8f741fb4a58bb060901d4359e4f975ca64bfa184da4e892c10cd"
)
EXTENDED_UNION_SHA256 = (
    "d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6"
)
SEED479_RECEIPTS = {
    "docs/data/seed479-population-ablation/receipt.json": "f7c5edd6e6bedd496b145721edcd77196ccdb32e127340c81b1546a6db816314",
    "docs/data/seed479-population-ablation/registration.json": "4e2a03af089593f1ae32abc26f8a407bb5d980be7100776c091224a7ae0b2fb4",
    "docs/data/seed479-population-ablation/evidence.json": "9056f7654b1dca824eccc783c04b7ad1a68140667e28068f665d304eea9ce2fe",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _points(winner: str) -> float:
    if winner in ("challenger", "F"):
        return 1.0
    if winner == "draw":
        return 0.5
    if winner in ("current", "initializer"):
        return 0.0
    raise ValueError("outcome_unknown")


def _replay_game(
    opening: dict[str, Any], row: dict[str, Any], seat: int
) -> tuple[str, int, int]:
    """Replay relative opening prefixes and absolute arena trajectory indices."""
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list):
        raise ValueError("opening_prefix_missing")
    for move in prefix:
        if game.over() or int(move) not in game.possible_moves():
            raise ValueError("opening_prefix_replay")
        if not game.move(game.pit_index(int(move))):
            raise ValueError("opening_prefix_replay")
    if game.to_state() != opening.get("state"):
        raise ValueError("opening_prefix_state")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("trajectory_missing")
    moves = [int(value) for value in trajectory.split(",") if value != ""]
    if len(moves) > 200 or int(row.get("game_length", -1)) != len(moves):
        raise ValueError("trajectory_length_or_cap")
    for ply, absolute in enumerate(moves):
        relative = absolute - game.current_player * 6
        if (
            game.over()
            or not 0 <= absolute < 12
            or absolute // 6 != game.current_player
            or relative not in game.possible_moves()
        ):
            raise ValueError(f"trajectory_illegal:{ply}")
        acting_player = game.current_player
        if not game.move(absolute):
            raise ValueError(f"trajectory_replay:{ply}")
        # Extra turns are represented by the unchanged current_player, not by
        # alternating seats in the recorded action stream.
        if game.current_player not in (acting_player, 1 - acting_player):
            raise ValueError("trajectory_turn_transition")
    if game.over() and len(moves) < 200:
        pass
    elif len(moves) != 200:
        raise ValueError("trajectory_ended_before_terminal_or_cap")
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("winner") != winner or int(row.get("margin", 10**9)) != margin:
        raise ValueError("trajectory_outcome_mismatch")
    return winner, margin, len(moves)


def _verify_artifact_against_frozen_state(
    root: Path, out: Path, arm: str, manifest: dict[str, str]
) -> None:
    """Compare exports to frozen states using read-only inputs and no file writes."""
    import torch

    from ml.alphazero_lite import train

    artifact = out / "runtime-artifacts" / arm
    if arm == "initializer":
        frozen_path = (
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, frozen_path)
        expected = train.checkpoint_from_model(model)
    else:
        trajectory = (
            root / "docs/data/seed479-population-ablation/trajectory-evidence.npz"
        )
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        parameters = list(model.parameters())
        with np.load(trajectory, allow_pickle=False) as archive:
            for index, parameter in enumerate(parameters):
                source = archive[f"F_step_16_post_parameter_{index:02d}"]
                if tuple(source.shape) != tuple(parameter.shape):
                    raise ValueError("frozen_F_parameter_shape")
                parameter.data.copy_(torch.from_numpy(source.copy()))
        expected = train.checkpoint_from_model(model)
    with np.load(artifact / "model.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f"artifact_model_keys:{arm}")
        for key, value in expected.items():
            if not np.array_equal(saved[key], value):
                raise ValueError(f"artifact_model_state:{arm}:{key}")
    weights = json.loads((artifact / "weights.json").read_text(encoding="utf-8"))
    if set(weights) != set(expected):
        raise ValueError(f"artifact_weights_keys:{arm}")
    for key, value in expected.items():
        if not np.array_equal(np.asarray(weights[key], dtype=np.float32), value):
            raise ValueError(f"artifact_weights_state:{arm}:{key}")
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    artifact_bindings = metadata.get("artifacts", {})
    if (
        metadata.get("schema_version") != "azlite_model_v1"
        or metadata.get("version") != f"seed488-{arm}"
        or metadata.get("game") != "kalah"
        or metadata.get("rules_version") != "kalah_v1"
        or metadata.get("input_encoding") != "kalah_v3"
        or metadata.get("feature_count") != 27
        or metadata.get("policy_size") != 6
        or metadata.get("architecture", {}).get("model_type") != "residual_v3"
        or artifact_bindings.get("weights_file") != "weights.json"
        or artifact_bindings.get("weights_fallback_file") != "model.npz"
        or artifact_bindings.get("weights_sha256") != _sha(artifact / "model.npz")
        or artifact_bindings.get("weights_json_sha256")
        != _sha(artifact / "weights.json")
    ):
        raise ValueError(f"artifact_metadata:{arm}")
    policy_path = root / "model-artifact/current/search_policy.json"
    if (artifact / "search_policy.json").read_bytes() != policy_path.read_bytes():
        raise ValueError(f"artifact_search_policy:{arm}")
    expected_files = {
        "model.npz",
        "weights.json",
        "metadata.json",
        "search_policy.json",
    }
    if set(manifest) != expected_files:
        raise ValueError(f"artifact_manifest_files:{arm}")


def _verify_supersession(out: Path) -> None:
    supersession = json.loads((out / "supersession.json").read_text(encoding="utf-8"))
    if (
        supersession.get("status")
        != "seed488_batch_identity_preparation_superseded_before_execution"
        or supersession.get("supersedes_pr") != 488
        or supersession.get("no_outcomes_existed_at_supersession") is not True
    ):
        raise ValueError("supersession_record")
    for original, binding in supersession.get("preserved_originals", {}).items():
        preserved = out / binding["preserved_as"]
        # The recorded mtime is historical provenance. Checkout/archive tools
        # may assign a different current mtime, so content identity is the
        # portable authenticity check.
        if _sha(preserved) != binding["sha256"]:
            raise ValueError(f"superseded_evidence_changed:{original}")


def _superseded_file_snapshot(out: Path) -> dict[Path, tuple[bytes, int]]:
    supersession_path = out / "supersession.json"
    if not supersession_path.is_file():
        return {}
    supersession = json.loads(supersession_path.read_text(encoding="utf-8"))
    snapshot = {
        out / binding["preserved_as"]: (
            (out / binding["preserved_as"]).read_bytes(),
            (out / binding["preserved_as"]).stat().st_mtime_ns,
        )
        for binding in supersession.get("preserved_originals", {}).values()
    }
    provenance_path = out / "failure-provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for path in (provenance_path, out / EXECUTION_FREEZE):
            if path.is_file():
                snapshot[path] = (path.read_bytes(), path.stat().st_mtime_ns)
        for relative in provenance.get("preserved_files", {}):
            path = out / relative
            snapshot[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    return snapshot


def _assert_superseded_snapshot_unchanged(
    snapshot: dict[Path, tuple[bytes, int]],
) -> None:
    for path, (content, mtime_ns) in snapshot.items():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"verification_mutated_superseded_evidence:{path.name}")
        if path.stat().st_mtime_ns != mtime_ns:
            raise ValueError(f"verification_changed_superseded_mtime:{path.name}")


def _expected_batch_runtime_contract(
    registration: dict[str, Any],
    opening_index: int,
    artifact_manifest_sha256: str,
    batch_sha256: str,
    entry_state_hash: str,
) -> dict[str, Any]:
    identities = registration["runtime"]["identities_sha256"]
    search_options = dict(registration["runtime"]["search_options"])
    search_options.pop("random_opening_plies", None)
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
        "batch_suite_sha256": batch_sha256,
        "parent_opening_index": opening_index,
        "parent_entry_state_hash": entry_state_hash,
        "local_opening_index": 0,
        "seed_derivation": "488 + parent_opening_index",
        "max_moves": 200,
        "opening_configuration": {"random_opening_plies": 0},
        "challenger_search_options": search_options,
        "current_search_options": search_options,
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
        "batch_artifact_dir": "resume-corrected",
    }


def _verify_execution_freeze(
    root: Path, out: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    """Authenticate the corrected, review-required freeze without rewriting evidence."""
    freeze = json.loads((out / EXECUTION_FREEZE).read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "seed488-pre-game-execution-freeze-v1"
        or freeze.get("status") != "frozen_before_execution_review_required"
        or freeze.get("supersedes_pr") != 491
        or freeze.get("execution_namespace") != "resume-corrected"
        or freeze.get("registration_sha256") != _sha(out / "registration.json")
        or freeze.get("receipt_sha256") != _sha(out / "receipt.json")
        or freeze.get("opening_suite_sha256") != _sha(out / "openings.jsonl")
    ):
        raise ValueError("execution_freeze_identity")
    expected_overrides = dict(registration["runtime"]["search_options"])
    expected_overrides.pop("random_opening_plies", None)
    if (
        freeze.get("opening_configuration") != {"random_opening_plies": 0}
        or freeze.get("per_side_search_options")
        != {"challenger": expected_overrides, "current": expected_overrides}
        or registration["runtime"]["search_options"].get("random_opening_plies") != 0
    ):
        raise ValueError("execution_freeze_option_partition")
    provenance_path = out / "failure-provenance.json"
    if _sha(provenance_path) != freeze.get("failure_provenance_sha256"):
        raise ValueError("failure_provenance_binding")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "seed488_attempt_491_failed_before_game"
        or provenance.get("completed_games") != 0
        or provenance.get("outcomes_committed") is not False
    ):
        raise ValueError("failure_provenance_status")
    for relative, digest in provenance.get("preserved_files", {}).items():
        if _sha(out / relative) != digest:
            raise ValueError(f"failed_attempt_evidence_binding:{relative}")
    for relative, digest in freeze.get("source_sha256", {}).items():
        actual = _sha(root / relative)
        if actual == digest:
            continue
        correction_path = out / "opening-0000-recovery-receipt.json"
        if not correction_path.is_file():
            raise ValueError(f"execution_freeze_source_binding:{relative}")
        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        if relative == "ml/alphazero_lite/test_seed488_strength_evaluation.py":
            if correction.get("historical_execution", {}).get(
                "source_sha256"
            ) != freeze.get("source_sha256"):
                raise ValueError(f"historical_source_binding:{relative}")
            continue
        prospective = correction.get("prospective_correction", {}).get(
            "source_sha256", {}
        )
        source_bound = prospective.get(relative) == actual
        continuation_path = out / "opening-0000-continuation-binding-v4.json"
        if not source_bound and continuation_path.is_file():
            continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
            source_bound = (
                continuation.get("recovery_receipt_sha256") == _sha(correction_path)
                and continuation.get("source_sha256", {}).get(relative) == actual
            )
        if (
            correction.get("historical_execution", {}).get("source_sha256")
            != freeze.get("source_sha256")
            or not source_bound
        ):
            raise ValueError(f"prospective_source_binding:{relative}")
    manifest_path = out / "runtime-artifacts.json"
    if _sha(manifest_path) != freeze.get("runtime_artifact_manifest_sha256"):
        raise ValueError("execution_freeze_artifact_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != freeze.get("runtime_artifacts_sha256"):
        raise ValueError("execution_freeze_artifact_identities")
    for arm, artifacts in manifest.items():
        for name, digest in artifacts.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"execution_freeze_artifact:{arm}:{name}")
        _verify_artifact_against_frozen_state(root, out, arm, artifacts)
    failed_opening = out / "resume/opening-0000.jsonl"
    rows = _rows(out / "openings.jsonl")
    if (
        failed_opening.read_bytes()
        != (json.dumps(rows[0], sort_keys=True, separators=(",", ":")) + "\n").encode()
        or _sha(failed_opening) != freeze.get("failed_batch_sha256")
        or (out / "resume/games-0000.jsonl").exists()
        or (out / "resume/report-0000.json").exists()
        or (out / "outcomes.jsonl.pending").exists()
    ):
        raise ValueError("failed_attempt_batch_state")
    if registration.get("fresh_confirmation_authorized") is not False:
        raise ValueError("execution_freeze_confirmation_gate")
    return freeze


def _verify_readonly(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    suite_path = out / "openings.jsonl"
    receipt_path = out / "execution-receipt.json"
    if not receipt_path.is_file():
        receipt_path = out / "receipt.json"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    suite_digest = _sha(suite_path)
    if registration.get("schema") != "seed488-frozen-strength-evaluation-v2":
        raise ValueError("registration_schema")
    if registration.get("status") != "frozen_before_execution":
        raise ValueError("registration_not_frozen")
    if registration.get("fresh_confirmation_authorized") is not False:
        raise ValueError("confirmation_gate_open")
    if registration.get("supersedes") != (
        "PR #488 preparation registration with parent-suite digest passed to single-opening arena batches; this is a preparation-only correction before any outcomes"
    ):
        raise ValueError("supersession_declaration")
    runtime = registration.get("runtime", {})
    expected_search_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
        "root_prior_transform": None,
        "random_opening_plies": 0,
    }
    if (
        runtime.get("simulations_per_move") != 1200
        or runtime.get("c_puct") != 1.25
        or runtime.get("workers") != 1
        or runtime.get("max_moves") != 200
        or runtime.get("max_moves_policy")
        != "At 200 decision plies, stop without sweeping remaining pits; compare captured stores at the cap. A game ending earlier uses Kalah terminal sweeping. Recorded trajectory length must be terminal before 200 or exactly 200."
        or runtime.get("search_options") != expected_search_options
        or runtime.get("seed_contract")
        != "azlite_eval_seed_v2; batch-local opening index is 0; base seed = 488 + parent opening index; arena deterministically binds the single batch opening state, seat, game-within-opening, and ply"
    ):
        raise ValueError("runtime_protocol_mismatch")
    suite_contract = registration.get("opening_suite", {})
    if (
        suite_contract.get("generator_seed") != SEED
        or suite_contract.get("count") != OPENINGS
        or suite_contract.get("games_per_opening") != 2
        or suite_contract.get("game_count") != OPENINGS * 2
        or suite_contract.get("eligible_against_reconstructable_exclusions") is not True
        or "non-overlap with unresolved seed461 continuation population cannot be certified"
        not in suite_contract.get("exclusions", {}).get("limitation", "")
    ):
        raise ValueError("opening_protocol_mismatch")
    if registration.get("bootstrap") != {
        "unit": "512 opening clusters; resample with replacement and carry both seat games together",
        "replicates": REPLICATES,
        "seed": SEED,
        "advance_rule": "advance only if one-sided 95% percentile lower score bound > 0.50",
        "two_sided_interval": "95% percentile interval also reported descriptively",
    }:
        raise ValueError("bootstrap_protocol_mismatch")
    if (
        registration.get("primary_score")
        != "F wins + 0.5 * draws, divided by all 1,024 games"
        or registration.get("promotion") != "No promotion from this experiment alone."
        or registration.get("seed479_and_480_classification")
        != "promising exploratory results; preserved as exploratory"
        or registration.get("interpretation")
        != "exploratory direct strength evaluation on a newly registered suite; not fresh confirmation or an independently untouched holdout"
    ):
        raise ValueError("decision_protocol_mismatch")
    if registration.get("trajectory_contract") != (
        "arena.py at the runtime source hash; opening prefixes and trajectories are player-relative v2 moves converted by KalahGame.pit_index at each ply; rules use KalahGame.move including extra turns and terminal sweep; stored margin is challenger captured store minus current captured store; winner is the sign of that margin"
    ):
        raise ValueError("trajectory_contract")
    if registration["opening_suite"].get("opening_suite_sha256") != suite_digest:
        raise ValueError("suite_digest")
    batch_contract = registration["opening_suite"].get("batch_identity_contract", {})
    if (
        batch_contract.get("parent_suite_sha256") != suite_digest
        or batch_contract.get("parent_opening_index")
        != "opening_index in openings.jsonl; arena-local index is 0"
        or batch_contract.get("parent_entry_identity")
        != "state_hash of the parent suite entry, validated against exact batch bytes"
        or batch_contract.get("seed_derivation")
        != "registered base seed = 488 + parent_opening_index; batch-local index does not affect seed"
    ):
        raise ValueError("batch_identity_registration")
    execution_freeze_path = out / EXECUTION_FREEZE
    if execution_freeze_path.is_file():
        _verify_execution_freeze(root, out, registration)
    else:
        impl = root / "ml/alphazero_lite/seed488_strength_evaluation.py"
        if registration.get("implementation_sha256") != _sha(impl):
            raise ValueError("implementation_binding")
        if registration.get("independent_verifier_sha256") != _sha(
            root / "ml/alphazero_lite/verify_seed488_strength_evaluation.py"
        ):
            raise ValueError("verifier_binding")
        if registration.get("synthetic_tests_sha256") != _sha(
            root / "ml/alphazero_lite/test_seed488_strength_evaluation.py"
        ):
            raise ValueError("synthetic_test_binding")
    for relative, digest in registration.get("generator_dependency_sha256", {}).items():
        if _sha(root / relative) != digest:
            raise ValueError(f"generator_dependency_binding:{relative}")
    expected_runtime = registration["runtime"].get("identities_sha256", {})
    if not expected_runtime:
        raise ValueError("runtime_identity_inventory")
    if registration["runtime"].get("python_version") != sys.version:
        raise ValueError("python_runtime_version")
    if registration["runtime"].get("numpy_version") != np.__version__:
        raise ValueError("numpy_runtime_version")
    for relative, digest in expected_runtime.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"runtime_binding:{relative}")
    f_endpoint = registration["arms"]["F"]
    trajectory_path = root / f_endpoint["checkpoint_source"]
    if _sha(trajectory_path) != f_endpoint["trajectory_sha256"]:
        raise ValueError("f_trajectory_binding")
    if (
        _sha(
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        != registration["arms"]["initializer"]["checkpoint_sha256"]
    ):
        raise ValueError("initializer_binding")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        parameter_arrays = [
            archive[f"F_step_16_post_parameter_{i:02d}"] for i in range(22)
        ]
    flattened = np.concatenate([x.reshape(-1).astype("<f4") for x in parameter_arrays])
    if (
        hashlib.sha256(flattened.tobytes()).hexdigest()
        != f_endpoint["parameter_state_sha256"]
    ):
        raise ValueError("f_parameter_identity")
    if registration["arms"].get("seed479_receipts") != {
        "receipt_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/receipt.json"
        ],
        "registration_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/registration.json"
        ],
        "evidence_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/evidence.json"
        ],
    }:
        raise ValueError("seed479_receipt_binding")
    for relative, digest in SEED479_RECEIPTS.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed479_receipt_source:{relative}")
    openings = _rows(suite_path)
    if len(openings) != OPENINGS or [int(x["opening_index"]) for x in openings] != list(
        range(OPENINGS)
    ):
        raise ValueError("opening_coverage")
    if any(
        row.get("opening_contract") != "arena_player_relative_v2" for row in openings
    ):
        raise ValueError("opening_contract")
    if all(
        row.get("opening_contract") == "arena_player_relative_v2" for row in openings
    ):
        if (
            _sha(root / "docs/data/seed482-exclusion-manifest.json.gz")
            != "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d"
        ):
            raise ValueError("base_exclusion_file_hash")
        if (
            _sha(root / "docs/data/seed483-historical-probe-identities.json.gz")
            != "cb11ea5912c530f4a6c01a913b9e5083c454f639d24669a4675ef6d1c2e45c9b"
        ):
            raise ValueError("reconstructed_exclusion_file_hash")
        extended_path = root / "docs/data/seed484-extended-probe-union.json.gz"
        if _sha(extended_path) != EXTENDED_EXCLUSION_SHA256:
            raise ValueError("extended_exclusion_file_hash")
        excluded_base = json.loads(
            gzip.decompress(
                (root / "docs/data/seed482-exclusion-manifest.json.gz").read_bytes()
            )
        )
        excluded_add = json.loads(
            gzip.decompress(
                (
                    root / "docs/data/seed483-historical-probe-identities.json.gz"
                ).read_bytes()
            )
        )
        excluded_canonical = {
            row["canonical_identity"] for row in excluded_base["identities"]
        }
        excluded_canonical.update(
            row["canonical_identity"] for row in excluded_add["identities"]
        )
        excluded_float32 = {
            row["float32_identity"] for row in excluded_base["identities"]
        }
        excluded_float32.update(
            row["float32_identity"] for row in excluded_add["identities"]
        )
        extended = json.loads(gzip.decompress(extended_path.read_bytes()))
        if (
            extended.get("status") != "incomplete_seed461_replay_gap"
            or extended.get("final_union_pair_count") != 149_756
            or extended.get("final_union_pairs_sha256") != EXTENDED_UNION_SHA256
        ):
            raise ValueError("extended_exclusion_union_binding")
        reconstructed_pairs = {
            (row["canonical_identity"], row["float32_identity"])
            for row in excluded_base["identities"]
        }
        reconstructed_pairs.update(
            (row["canonical_identity"], row["float32_identity"])
            for row in excluded_add["identities"]
        )
        reconstructed_pairs.update(
            (row["canonical_identity"], row["float32_identity"])
            for row in extended["seed467_additional_identities"]
        )
        pair_payload = json.dumps(
            sorted([list(pair) for pair in reconstructed_pairs]), separators=(",", ":")
        ).encode()
        if (
            len(reconstructed_pairs) != 149_756
            or hashlib.sha256(pair_payload).hexdigest() != EXTENDED_UNION_SHA256
            or suite_contract.get("exclusions", {}).get("seed484_extended_union_sha256")
            != EXTENDED_UNION_SHA256
            or suite_contract.get("exclusions", {}).get(
                "seed484_extended_union_file_sha256"
            )
            != EXTENDED_EXCLUSION_SHA256
        ):
            raise ValueError("extended_exclusion_union_reconstruction")
        excluded_canonical.update(
            pair["canonical_identity"]
            for pair in extended["seed467_additional_identities"]
        )
        excluded_float32.update(
            pair["float32_identity"]
            for pair in extended["seed467_additional_identities"]
        )
        seen_states: set[str] = set()
        seen_float32: set[str] = set()
        for index, row in enumerate(openings):
            game = KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            absolute_prefix = row.get("source_prefix_moves_absolute")
            if not isinstance(absolute_prefix, list) or len(absolute_prefix) != len(
                row["prefix_moves"]
            ):
                raise ValueError("opening_absolute_prefix")
            for relative, absolute in zip(
                row["prefix_moves"], absolute_prefix, strict=True
            ):
                if int(absolute) != game.pit_index(int(relative)):
                    raise ValueError("opening_prefix_coordinate_mismatch")
                if relative not in game.possible_moves() or not game.move(
                    game.pit_index(int(relative))
                ):
                    raise ValueError("opening_prefix_replay")
            state = game.to_state()
            canonical = hashlib.sha256(
                json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            float_identity = hashlib.sha256(
                np.asarray(
                    encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
                ).tobytes()
            ).hexdigest()
            if state != row["state"] or canonical != row["state_hash"]:
                raise ValueError(f"opening_state_mismatch:{index}")
            if canonical in excluded_canonical or float_identity in excluded_float32:
                raise ValueError(f"opening_exclusion_overlap:{index}")
            if canonical in seen_states or float_identity in seen_float32:
                raise ValueError("opening_duplicate_state")
            seen_states.add(canonical)
            seen_float32.add(float_identity)
    if not receipt_path.exists():
        return {"status": "prepared_not_executed", "opening_count": len(openings)}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    partial_recovery = False
    if receipt.get("status") == "seed488_prepared_not_executed":
        for relative, digest in receipt.items():
            if relative != "status" and _sha(out / relative) != digest:
                raise ValueError(f"prepared_receipt_mismatch:{relative}")
        _verify_supersession(out)
        if execution_freeze_path.is_file():
            _verify_execution_freeze(root, out, registration)
            if not (out / "outcomes.jsonl").is_file():
                return {
                    "status": "ready_for_review_not_executed",
                    "opening_count": len(openings),
                    "completed_games": 0,
                }
            partial_recovery = True
        else:
            return {"status": "prepared_not_executed", "opening_count": len(openings)}
    elif receipt.get("status") != "seed488_strength_evaluation_complete":
        raise ValueError("completion_receipt_status")
    if not partial_recovery:
        for relative, digest in receipt.items():
            if relative != "status" and _sha(out / relative) != digest:
                raise ValueError(f"receipt_mismatch:{relative}")
        _verify_supersession(out)
    artifact_manifest = json.loads(
        (out / "runtime-artifacts.json").read_text(encoding="utf-8")
    )
    for arm, artifacts in artifact_manifest.items():
        for name, digest in artifacts.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"artifact_mismatch:{arm}:{name}")
        _verify_artifact_against_frozen_state(root, out, arm, artifacts)
    results_path = out / "outcomes.jsonl"
    rows = _rows(results_path)
    if (out / "outcomes.jsonl.pending").exists():
        raise ValueError("conflicting_pending_outcome_batch")
    recovery_receipt_path = out / "opening-0000-recovery-receipt.json"
    if (out / EXECUTION_FREEZE).is_file() and any(
        int(row.get("opening_index", -1)) == 0 for row in rows
    ):
        from ml.alphazero_lite.seed488_replay_recovery import verify_receipt

        recovery_receipt = verify_receipt(root)
        receipt_hash = _sha(recovery_receipt_path)
        opening_zero = [row for row in rows if int(row["opening_index"]) == 0]
        if len(opening_zero) != 2 or any(
            row.get("recovery_receipt_sha256") != receipt_hash
            or row.get("historical_execution_provenance")
            != recovery_receipt["historical_execution"]
            for row in opening_zero
        ):
            raise ValueError("recovery_continuation_provenance")
    expected = {(opening, seat) for opening in range(OPENINGS) for seat in (0, 1)}
    observed: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["opening_index"]), int(row["F_player"]))
        if key not in expected or key in observed:
            raise ValueError("duplicate_or_invalid_game")
        if row.get("opening_state_hash") != openings[key[0]]["state_hash"]:
            raise ValueError("game_opening_identity_mismatch")
        opening = key[0]
        artifact_dir = row.get("batch_artifact_dir")
        if artifact_dir != "resume-corrected":
            raise ValueError("batch_artifact_namespace")
        game_log = out / artifact_dir / f"games-{opening:04d}.jsonl"
        report = out / artifact_dir / f"report-{opening:04d}.json"
        opening_file = out / artifact_dir / f"opening-{opening:04d}.jsonl"
        if _sha(game_log) != row.get("batch_game_log_sha256"):
            raise ValueError("game_log_receipt_mismatch")
        if _sha(report) != row.get("batch_report_sha256"):
            raise ValueError("arena_report_receipt_mismatch")
        expected_opening_bytes = (
            json.dumps(openings[opening], sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if expected_opening_bytes != opening_file.read_bytes() or _sha(
            opening_file
        ) != row.get("batch_opening_sha256"):
            raise ValueError("paired_opening_receipt_mismatch")
        batch_rows = _rows(game_log)
        position = int(row["batch_game_position"])
        if len(batch_rows) != 2 or position not in (0, 1):
            raise ValueError("arena_batch_coverage")
        challenger_seats = [
            int(item.get("challenger_player", -1)) for item in batch_rows
        ]
        if sorted(challenger_seats) != [0, 1]:
            raise ValueError("arena_batch_seat_pairing")
        if any(
            item.get("opening_state_hash") != openings[opening]["state_hash"]
            or item.get("opening_prefix_moves", openings[opening]["prefix_moves"])
            != openings[opening]["prefix_moves"]
            for item in batch_rows
        ):
            raise ValueError("arena_batch_opening_identity")
        arena_row = batch_rows[position]
        arena_winner = arena_row.get("winner")
        if arena_winner == "challenger":
            arena_winner = "F"
        elif arena_winner == "current":
            arena_winner = "initializer"
        if int(arena_row.get("challenger_player", -1)) != key[
            1
        ] or arena_winner != row.get("winner"):
            raise ValueError("arena_game_outcome_mismatch")
        expected_runtime_contract = _expected_batch_runtime_contract(
            registration,
            opening,
            _sha(out / "runtime-artifacts.json"),
            _sha(opening_file),
            openings[opening]["state_hash"],
        )
        if row.get("batch_runtime_contract") != expected_runtime_contract:
            raise ValueError("batch_runtime_contract_mismatch")
        _replay_game(openings[opening], arena_row, key[1])
        report_data = json.loads(report.read_text(encoding="utf-8"))
        report_notes = report_data.get("notes", {})
        batch_outcomes = [str(game.get("winner")) for game in batch_rows]
        if (
            report_data.get("games_played") != 2
            or report_data.get("wins") != batch_outcomes.count("challenger")
            or report_data.get("losses") != batch_outcomes.count("current")
            or report_data.get("draws") != batch_outcomes.count("draw")
            or report_notes.get("challenger_simulations") != 1200
            or report_notes.get("current_simulations") != 1200
            or report_notes.get("workers_used") != 1
            or report_notes.get("random_opening_plies") != 0
            or report_notes.get("seed_contract") != "azlite_eval_seed_v2"
            or report_notes.get("base_seed") != SEED + opening
            or report_notes.get("suite_sha256") != _sha(opening_file)
            or report_notes.get("search_profile", {}).get("c_puct") != 1.25
            or report_notes.get("exact_root_solve_threshold") != 16
            or report_notes.get("exact_root_native_probe_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/runtime/kalah_v1_tablebase"
            ]
            or report_notes.get("exact_root_tablebase_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/runtime/kalah_v1_21.kvtb"
            ]
            or report_notes.get("runtime_search_policy_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/current/search_policy.json"
            ]
            or report_notes.get("runtime_search_policy_mode")
            != "puct_exact_root_hybrid"
            or report_notes.get("exact_root_objective") != "final_score_margin"
            or report_notes.get("exact_root_tie_rule")
            != "highest_legal_network_prior_then_lowest_move_index"
            or report_notes.get("exact_leaf_solve_mode") != "disabled"
            or report_notes.get("search_options")
            != {
                key: value
                for key, value in registration["runtime"]["search_options"].items()
                if key not in {"random_opening_plies", "root_prior_transform"}
            }
            or report_notes.get("seed") != SEED + opening
        ):
            raise ValueError("arena_report_runtime_contract")
        observed[key] = _points(str(row["winner"]))
    completed_openings = {opening for opening, _seat in observed}
    if any(
        (opening, 0) not in observed or (opening, 1) not in observed
        for opening in completed_openings
    ):
        raise ValueError("incomplete_opening_pair")
    if observed.keys() != expected:
        if (out / "results.json").exists():
            raise ValueError("premature_partial_results")
        return {
            "status": "verified_incomplete",
            "game_count": len(observed),
            "opening_count": len(completed_openings),
            "completed_openings": len(completed_openings),
            "remaining_openings": OPENINGS - len(completed_openings),
            "remaining_games": (OPENINGS - len(completed_openings)) * 2,
            "promotion": False,
        }
    clusters = np.asarray(
        [(observed[(i, 0)] + observed[(i, 1)]) / 2 for i in range(OPENINGS)],
        dtype=np.float64,
    )
    score = float(clusters.mean())
    rng = np.random.default_rng(SEED)
    draw_index = rng.integers(0, OPENINGS, size=(REPLICATES, OPENINGS))
    bootstrap_scores = clusters[draw_index].mean(axis=1)
    lower = float(np.quantile(bootstrap_scores, 0.05))
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    expected_result = {
        "game_count": OPENINGS * 2,
        "score": score,
        "bootstrap": {
            "method": "opening-paired percentile bootstrap; resample 512 opening clusters, carrying both seats together",
            "replicates": REPLICATES,
            "seed": SEED,
            "confidence_level": 0.95,
            "one_sided_lower": lower,
            "two_sided_interval": [
                float(np.quantile(bootstrap_scores, 0.025)),
                float(np.quantile(bootstrap_scores, 0.975)),
            ],
        },
        "advance": lower > 0.5,
        "promotion": False,
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            if key == "score" and np.isclose(
                result.get(key), value, rtol=0, atol=1e-15
            ):
                continue
            raise ValueError(f"result_mismatch:{key}")
    return {"status": "verified_complete", **expected_result}


def verify(root: Path) -> dict[str, Any]:
    """Verify publication while proving historical evidence remains untouched."""
    out = root.resolve() / OUT
    snapshot = _superseded_file_snapshot(out)
    try:
        return _verify_readonly(root)
    finally:
        _assert_superseded_snapshot_unchanged(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
