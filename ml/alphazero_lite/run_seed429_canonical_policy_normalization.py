"""Seed429 experiment runner and immutable evidence publisher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Any

import gzip
import numpy as np
import torch

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import arena as arena_module
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train_one_epoch,
)
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.seed429_policy_normalization import (
    canonical_identity_from_encoded_state,
    normalize_policy_coefficients,
    reconstruct_frozen_census,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed429-canonical-policy-normalization"
EXECUTION_SOURCES = (
    "ml/alphazero_lite/run_seed429_canonical_policy_normalization.py",
    "ml/alphazero_lite/seed429_policy_normalization.py",
    "ml/alphazero_lite/verify_seed429_publication.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/report_validation.py",
    "ml/alphazero_lite/value_transforms.py",
    "ml/alphazero_lite/root_prior_transforms.py",
    "ml/alphazero_lite/policy_prior_localization.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/endgame_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/opening_cache.py",
    "ml/alphazero_lite/search_ablation.py",
    "ml/alphazero_lite/shadow_root_q.py",
    "ml/alphazero_lite/export_artifact.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/seed422_exclusions.py",
    "ml/alphazero_lite/seed416_policy_target_softening.py",
    "ml/alphazero_lite/seed427_validation_metrics.py",
    "ml/alphazero_lite/seed427_validation_subsets.py",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        if (DATA / "registration.json").exists():
            raise ValueError(f"immutable_evidence_conflict:{path.name}")
    if not path.exists() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")


def write_progress(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_artifact_files(artifact: Path, expected: dict[str, str]) -> None:
    """Reject missing or altered runtime artifact files before evaluation."""
    for filename, digest in expected.items():
        path = artifact / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"seed429_artifact_identity_mismatch:{filename}")


def verify_execution_snapshots(
    root: Path, snapshots: dict[str, str], source_names: tuple[str, ...]
) -> None:
    """Require a complete snapshot map and exact source-to-snapshot byte identity."""
    expected_paths = {
        f"docs/data/seed429-canonical-policy-normalization/execution-source-snapshots/{Path(name).name}"
        for name in source_names
    }
    if set(snapshots) != expected_paths:
        raise ValueError("seed429_execution_source_inventory_incomplete")
    for source_name in source_names:
        snapshot_relative = (
            "docs/data/seed429-canonical-policy-normalization/"
            f"execution-source-snapshots/{Path(source_name).name}"
        )
        snapshot_path = root / snapshot_relative
        source_path = root / source_name
        expected_hash = snapshots[snapshot_relative]
        if (
            not snapshot_path.is_file()
            or not source_path.is_file()
            or sha256_file(snapshot_path) != expected_hash
            or sha256_file(source_path) != expected_hash
        ):
            raise ValueError(
                f"seed429_execution_source_snapshot_mismatch:{source_name}"
            )


def census() -> dict[str, Any]:
    evidence = reconstruct_frozen_census(ROOT)
    write_json(DATA / "coefficient-census.json", evidence)
    if evidence["normalization_is_noop"]:
        write_json(
            DATA / "pretraining-stop.json",
            {
                "classification": "normalization is a no-op",
                "census_sha256": __import__("hashlib")
                .sha256((DATA / "coefficient-census.json").read_bytes())
                .hexdigest(),
                "training_started": False,
                "arena_started": False,
            },
        )
    return evidence


def exclusion_union() -> tuple[set[str], dict[str, Any]]:
    """Rebuild the #416–421 proof and add strictly replayed #422 trajectories."""
    from ml.alphazero_lite.seed422_exclusions import (
        build_union as build_422_historical_union,
        replay_game,
        replay_opening,
        rows,
    )

    historical, historic_proof = build_422_historical_union()
    seed422 = ROOT / "docs/data/seed422-adam-first-moment"
    suite_path = seed422 / "openings.jsonl"
    ledger_path = seed422 / "outcome-ledger.jsonl"
    suite_rows = rows(suite_path)
    ledger_rows = rows(ledger_path)
    if len(suite_rows) != 512 or len(ledger_rows) != 2048:
        raise ValueError("seed422_completed_arena_evidence_incomplete")
    consumed: set[str] = set()
    roots: set[str] = set()
    for opening in suite_rows:
        game = replay_opening(opening)
        identity = suites.canonical_key(game.to_state())
        if identity != suites.canonical_key(opening["state"]):
            raise ValueError("seed422_declared_opening_replay_mismatch")
        roots.add(identity)
    seen_games: set[tuple[str, int, int]] = set()
    for record in ledger_rows:
        game_record = record["game"]
        opening_id = int(record["opening_id"])
        key = (
            str(record["lane"]),
            opening_id,
            int(game_record["challenger_player"]),
        )
        if key in seen_games or not 0 <= opening_id < len(suite_rows):
            raise ValueError("seed422_duplicate_or_invalid_arena_outcome")
        seen_games.add(key)
        consumed.update(replay_game(suite_rows[opening_id], game_record))
    expected = {
        (lane, opening, seat)
        for lane in ("A", "B")
        for opening in range(512)
        for seat in (0, 1)
    }
    if seen_games != expected:
        raise ValueError("seed422_arena_seat_pairing_incomplete")
    all_excluded = historical | roots | consumed
    components = [
        {
            "name": "seed422-recomputed-historical-union-through-421",
            "path": "docs/data/seed422-adam-first-moment/opening-exclusion-proof.json",
            "sha256": sha256_file(seed422 / "opening-exclusion-proof.json"),
            "recomputed_identity_count": len(historical),
            "identity_set_sha256": historic_proof["identity_set_sha256"],
        },
        {
            "name": "seed422-declared-suite-roots",
            "path": "docs/data/seed422-adam-first-moment/openings.jsonl",
            "sha256": sha256_file(suite_path),
            "recomputed_identity_count": len(roots),
            "identity_set_sha256": hashlib.sha256(
                "\n".join(sorted(roots)).encode()
            ).hexdigest(),
        },
        {
            "name": "seed422-strictly-replayed-completed-arena-trajectories",
            "path": "docs/data/seed422-adam-first-moment/outcome-ledger.jsonl",
            "sha256": sha256_file(ledger_path),
            "recomputed_identity_count": len(consumed),
            "identity_set_sha256": hashlib.sha256(
                "\n".join(sorted(consumed)).encode()
            ).hexdigest(),
            "strict_complete_games": len(seen_games),
        },
    ]
    # #426–428 are retrospective over identities already present in this union's
    # registered #416 replay rows and the #422 completed evidence. Bind their
    # manifests/receipts explicitly rather than assuming this from row counts.
    for relative in (
        "docs/data/seed426-canonical-overlap/manifest.json",
        "docs/data/seed426-canonical-overlap/results.json",
        "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-manifest.json",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-results.json",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-prediction-evidence.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
        "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
        "docs/data/seed426-canonical-overlap/seed428-source-contributions.json",
        "docs/data/seed426-canonical-overlap/seed428-semantic-verification-receipt.json",
    ):
        path = ROOT / relative
        components.append(
            {
                "name": relative.rsplit("/", 1)[-1],
                "path": relative,
                "sha256": sha256_file(path),
                "coverage": "retrospective artifacts bound; their source states are replay-row identities already in the verified historical union",
            }
        )
    retrospective_reconciliation: dict[str, dict[str, int]] = {}
    for name, relative in (
        (
            "seed426-row-accounting",
            "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz",
        ),
        (
            "seed427-validation-membership",
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        ),
        (
            "seed427-model-evaluated-predictions",
            "docs/data/seed426-canonical-overlap/seed427-prediction-evidence.jsonl.gz",
        ),
    ):
        identities: set[str] = set()
        with gzip.open(ROOT / relative, "rt", encoding="utf-8") as stream:
            for row in (json.loads(line) for line in stream if line.strip()):
                membership = row.get("membership", row)
                encoded_identity = str(membership["canonical_identity"])
                fields = json.loads(encoded_identity)
                independently_encoded = json.dumps(
                    (
                        tuple(int(value) for value in fields[0]),
                        tuple(int(value) for value in fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                    ),
                    separators=(",", ":"),
                )
                if encoded_identity != independently_encoded:
                    raise ValueError(f"{name}_canonical_identity_encoding_invalid")
                state = {
                    "player_pits": fields[0],
                    "opponent_pits": fields[1],
                    "player_store": fields[2],
                    "opponent_store": fields[3],
                    "current_player": fields[4],
                }
                identities.add(suites.canonical_key(state))
        missing = identities - all_excluded
        retrospective_reconciliation[name] = {
            "identity_count": len(identities),
            "already_excluded": len(identities & all_excluded),
            "added_identity_count": len(missing),
        }
        components.append(
            {
                "name": name,
                "path": relative,
                "sha256": sha256_file(ROOT / relative),
                "declared_or_evaluated_identity_count": len(identities),
                "identities_already_in_preexisting_union": len(
                    identities & all_excluded
                ),
                "identities_added_to_union": len(missing),
                "identity_set_sha256": hashlib.sha256(
                    "\n".join(sorted(identities)).encode()
                ).hexdigest(),
                "missing_identity_sha256": hashlib.sha256(
                    "\n".join(sorted(missing)).encode()
                ).hexdigest(),
            }
        )
        all_excluded.update(missing)
    proof = {
        "schema": "seed429-opening-exclusion-proof-v1",
        "identity_encoding": "canonical Kalah player-relative ordered integer state hashed by build_opening_suite.canonical_key",
        "historical_union_through_421": historic_proof,
        "components": components,
        "retrospective_identity_reconciliation": {
            name: counts for name, counts in retrospective_reconciliation.items()
        },
        "historical_suite_root_count": len(roots),
        "strictly_replayed_consumed_identity_count": len(consumed),
        "combined_identity_count": len(all_excluded),
        "combined_identity_set_sha256": hashlib.sha256(
            "\n".join(sorted(all_excluded)).encode()
        ).hexdigest(),
        "excluded_identities": sorted(all_excluded),
        "seed422_game_accounting": {
            "games": len(seen_games),
            "expected": len(expected),
        },
        "seed416_validation_replay_state_membership": "all five seed416 replay sources verified and included in historical union; #426–428 retrospective state evidence is bound above",
    }
    return all_excluded, proof


def freeze_suite() -> tuple[Path, dict[str, Any]]:
    excluded, proof = exclusion_union()
    proof_path = DATA / "opening-exclusion-proof.json"
    write_json(proof_path, proof)
    population = suites.deduplicate_openings(suites.enumerate_legal_prefixes(8))[0]
    eligible = [
        row
        for row in population
        if int(row["pit_sum"]) > 32
        and not KalahGame.from_state(row["state"]).over()
        and suites.canonical_key(row["state"]) not in excluded
    ]
    selected = [
        suites.export_arena_entry(row)
        for row in suites.select_diverse(suites.stratify_openings(eligible), 512, 429)
    ]
    suite_path = DATA / "openings.jsonl"
    suite_payload = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in selected
    )
    if suite_path.exists() and suite_path.read_text(encoding="utf-8") != suite_payload:
        if (DATA / "registration.json").exists():
            raise ValueError("frozen_suite_conflict")
    if (
        not suite_path.exists()
        or suite_path.read_text(encoding="utf-8") != suite_payload
    ):
        suite_path.write_text(suite_payload, encoding="utf-8")
    suites.validate_arena_entries(suites.load_suite_jsonl(str(suite_path)))
    ids: set[str] = set()
    for row in selected:
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in row["prefix_moves"]]
        for relative_move in prefix:
            if (
                game.over()
                or relative_move not in game.possible_moves()
                or not game.move(game.pit_index(relative_move))
            ):
                raise ValueError("seed429_player_relative_prefix_replay_failed")
        identity = suites.canonical_key(game.to_state())
        if identity != suites.canonical_key(row["state"]):
            raise ValueError("seed429_opening_replayed_state_mismatch")
        if identity in ids or identity in excluded:
            raise ValueError("seed429_opening_identity_overlap")
        ids.add(identity)
        if game.over() or int(row["pit_sum"]) <= 32:
            raise ValueError("seed429_opening_ineligible")
    if len(ids) != 512:
        raise ValueError("seed429_opening_count_mismatch")
    return suite_path, proof


def _training_arrays() -> tuple[np.ndarray, ...]:
    registration = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text(encoding="utf-8")
    )
    # A targets are byte-bound #416 lane-A derivatives; the transforms are verified
    # by #416's public verifier and retain each source row's original target.
    derivative_paths = [
        Path(registration["derivatives"][record["name"]]["A"]["derivative"])
        for record in registration["replays"]
    ]
    weights = [int(row["weight"]) for row in registration["replays"]]
    x, policy, value, replay, coefficients = load_jsonl_replay(
        derivative_paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[
            row["value_target_mode"] for row in registration["replays"]
        ],
        include_policy_loss_weights=True,
    )
    split_path = ROOT / registration["training"]["source_row_split"]["path"]
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    validation_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    train_rows = set(int(row) for row in replay[train_positions])
    validation_rows = set(int(row) for row in replay[validation_positions])
    if train_rows & validation_rows or train_rows | validation_rows != set(
        range(len(x))
    ):
        raise ValueError("seed429_compact_train_validation_partition_invalid")
    identities = [canonical_identity_from_encoded_state(row.tolist()) for row in x]
    stones = np.rint(x[:, :12].astype(np.float64) * 48).sum(axis=1)
    compact_sources: list[str] = []
    for record in registration["replays"]:
        with Path(record["path"]).open(encoding="utf-8") as stream:
            compact_sources.extend(
                [record["name"]] * sum(1 for line in stream if line.strip())
            )
    if len(compact_sources) != len(x):
        raise ValueError("seed429_compact_source_mapping_invalid")
    training_mask = np.zeros(len(x), dtype=bool)
    training_mask[list(train_rows)] = True
    normalized, summary = normalize_policy_coefficients(
        coefficients, identities, stones, compact_sources, training_mask
    )
    census_record = json.loads((DATA / "coefficient-census.json").read_text())
    if (
        hashlib.sha256(np.asarray(coefficients, dtype="<f4").tobytes()).hexdigest()
        != census_record["control_coefficients_sha256"]
        or hashlib.sha256(np.asarray(normalized, dtype="<f4").tobytes()).hexdigest()
        != census_record["treatment_coefficients_sha256"]
    ):
        raise ValueError("seed429_coefficient_vector_binding_mismatch")
    if not np.array_equal(
        normalized[list(validation_rows)], coefficients[list(validation_rows)]
    ):
        raise ValueError("seed429_validation_coefficients_changed")
    if not np.array_equal(normalized[stones <= 16], coefficients[stones <= 16]):
        raise ValueError("seed429_low_stone_coefficients_changed")
    if summary["rows_changed"] != census_record["rows_changed"]:
        raise ValueError("seed429_census_changed_row_count_mismatch")
    return x, policy, value, replay, coefficients, normalized, train_positions


def train() -> dict[str, Any]:
    """Train both fixed arms with frozen permutations and epoch-level resume."""
    if not (DATA / "registration.json").is_file():
        raise ValueError("seed429_prospective_registration_missing")
    registration_hash = sha256_file(DATA / "registration.json")
    progress_dir = ROOT / ".tmp/seed429-canonical-policy-normalization"
    progress_dir.mkdir(parents=True, exist_ok=True)
    x, policy, value, replay, control, treatment, train_positions = _training_arrays()
    train_replay = replay[train_positions]
    freeze = ROOT / "docs/data/seed416-policy-target-softening/training-freeze-v3"
    with gzip.open(
        freeze / "epoch-permutations.json.gz", "rt", encoding="utf-8"
    ) as stream:
        permutations = json.load(stream)
    expected_a = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/training-results.json"
        ).read_text(encoding="utf-8")
    )["lanes"]["A"]["epochs"]
    init = Path(
        json.loads(
            (
                ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
            ).read_text(encoding="utf-8")
        )["seed455_initialization"]["path"]
    )
    result: dict[str, Any] = {}
    for lane, coefficients in (("A", control), ("B", treatment)):
        lane_dir = progress_dir / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        state_path = lane_dir / "resume.pt"
        progress_path = lane_dir / "progress.json"
        checkpoints = lane_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        if progress_path.exists() != state_path.exists():
            raise ValueError(f"seed429_resume_pair_incomplete:{lane}")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("registration_sha256") != registration_hash:
                raise ValueError(f"seed429_resume_registration_mismatch:{lane}")
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            state_progress = saved.get("progress")
            if (
                not isinstance(state_progress, dict)
                or state_progress.get("registration_sha256") != registration_hash
            ):
                raise ValueError(f"seed429_resume_state_progress_invalid:{lane}")
            if progress != state_progress:
                progress = state_progress
                write_progress(progress_path, progress)
            for epoch, digest in progress["epoch_hashes"].items():
                checkpoint = lane_dir / "checkpoints" / f"E{epoch}.npz"
                if not checkpoint.is_file() or sha256_file(checkpoint) != digest:
                    raise ValueError(
                        f"seed429_resume_checkpoint_mismatch:{lane}:E{epoch}"
                    )
            model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            model.load_state_dict(saved["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
            optimizer.load_state_dict(saved["optimizer"])
            generator = torch.Generator(device="cpu")
            generator.set_state(saved["generator_state"])
            start_epoch = int(progress["completed_epoch"]) + 1
            history = list(progress["history"])
            epoch_hashes = dict(progress["epoch_hashes"])
        else:
            set_seed(416)
            model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            load_checkpoint_into_model(model, init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
            generator = torch.Generator(device="cpu").manual_seed(416)
            start_epoch = 1
            history = []
            epoch_hashes = {}
        for epoch in range(start_epoch, 5):
            seen_permutation: list[list[int]] = []
            metrics = train_one_epoch(
                model=model,
                optimizer=optimizer,
                compact_x=x,
                compact_p=policy,
                compact_v=value,
                compact_policy_loss_weights=coefficients,
                replay_indexes=train_replay,
                batch_size=512,
                device=torch.device("cpu"),
                value_loss_weight=0.3,
                value_loss="huber",
                huber_delta=1.0,
                grad_clip=1.0,
                primary_order_generator=generator,
                permutation_callback=lambda _e, p: seen_permutation.append(p),
                epoch=epoch,
            )
            if (
                len(seen_permutation) != 1
                or seen_permutation[0] != permutations[epoch - 1]
            ):
                raise ValueError(f"seed429_frozen_permutation_mismatch:{lane}:E{epoch}")
            checkpoint = checkpoints / f"E{epoch}.npz"
            checkpoint_payload = checkpoint_from_model(model)
            if checkpoint.exists():
                with np.load(checkpoint) as existing:
                    if any(
                        not np.array_equal(existing[name], value)
                        for name, value in checkpoint_payload.items()
                    ):
                        raise ValueError(f"seed429_checkpoint_conflict:{lane}:E{epoch}")
            else:
                np.savez(checkpoint, **checkpoint_payload)
            epoch_hash = sha256_file(checkpoint)
            if lane == "A" and epoch_hash != expected_a[str(epoch)]:
                raise ValueError(f"seed429_A_reproduction_failed:E{epoch}")
            epoch_hashes[str(epoch)] = epoch_hash
            history.append({"epoch": epoch, **metrics})
            progress = {
                "registration_sha256": registration_hash,
                "completed_epoch": epoch,
                "history": history,
                "epoch_hashes": epoch_hashes,
            }
            state_staging = state_path.with_suffix(".pt.staging")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                    "progress": progress,
                },
                state_staging,
            )
            os.replace(state_staging, state_path)
            write_progress(progress_path, progress)
        if (
            lane == "A"
            and epoch_hashes.get("4")
            != "836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd"
        ):
            raise ValueError("seed429_A_E4_required_identity_mismatch")
        result[lane] = {
            "epoch_hashes": epoch_hashes,
            "history": history,
            "optimizer_updates": sum(
                int(row.get("optimizer_updates", 0)) for row in history
            ),
            "checkpoint_dir": str(checkpoints),
        }
    if any(result[lane]["optimizer_updates"] != 1052 for lane in ("A", "B")):
        raise ValueError("seed429_optimizer_update_count_mismatch")
    for lane in ("A", "B"):
        local_dir = Path(result[lane]["checkpoint_dir"])
        published_dir = DATA / "training-checkpoints" / lane
        published_dir.mkdir(parents=True, exist_ok=True)
        for epoch, expected in result[lane]["epoch_hashes"].items():
            src = local_dir / f"E{epoch}.npz"
            dst = published_dir / f"E{epoch}.npz"
            if not dst.exists():
                shutil.copy2(src, dst)
            if sha256_file(dst) != expected:
                raise ValueError(
                    f"seed429_checkpoint_publication_mismatch:{lane}:E{epoch}"
                )
        result[lane]["checkpoint_dir"] = str(published_dir.relative_to(ROOT))
    published = {
        "schema": "seed429-training-results-v1",
        "registration_sha256": registration_hash,
        "lanes": result,
    }
    write_json(DATA / "training-results.json", published)
    return published


def readiness(*, require_registration: bool = False) -> dict[str, Any]:
    """Check all prospective inputs and bindings without model training or games."""
    census_path = DATA / "coefficient-census.json"
    if not census_path.is_file():
        raise ValueError("seed429_published_census_missing")
    census_record = json.loads(census_path.read_text(encoding="utf-8"))
    reconstructed_census = reconstruct_frozen_census(ROOT)
    if reconstructed_census != census_record:
        raise ValueError("seed429_census_reconstruction_mismatch")
    if census_record["normalization_is_noop"]:
        raise ValueError("seed429_noop_must_stop_before_training")
    suite_path = DATA / "openings.jsonl"
    proof_path = DATA / "opening-exclusion-proof.json"
    for path in (suite_path, proof_path):
        if not path.is_file():
            raise ValueError(f"seed429_required_input_missing:{path.name}")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    rebuilt_excluded, rebuilt_proof = exclusion_union()
    if proof != rebuilt_proof or set(proof["excluded_identities"]) != rebuilt_excluded:
        raise ValueError("seed429_exclusion_proof_reconstruction_mismatch")
    final_ids = set(proof["excluded_identities"])
    if len(final_ids) != proof["combined_identity_count"]:
        raise ValueError("seed429_exclusion_identity_count_mismatch")
    for component in proof["components"]:
        component_path = ROOT / component["path"]
        if (
            not component_path.is_file()
            or sha256_file(component_path) != component["sha256"]
        ):
            raise ValueError(
                f"seed429_exclusion_component_binding_invalid:{component['name']}"
            )
    for name in (
        "seed426-row-accounting",
        "seed427-validation-membership",
        "seed427-model-evaluated-predictions",
    ):
        component = next(row for row in proof["components"] if row["name"] == name)
        if (
            component["identities_added_to_union"] != 0
            or component["identities_already_in_preexisting_union"]
            != component["declared_or_evaluated_identity_count"]
        ):
            raise ValueError(f"seed429_exclusion_coverage_incomplete:{name}")
    suite_rows = suites.load_suite_jsonl(str(suite_path))
    suite_ids: set[str] = set()
    for row in suite_rows:
        game = KalahGame.from_state(suites.INITIAL_STATE)
        moves = [int(move) for move in row["prefix_moves"]]
        for move in moves:
            if (
                game.over()
                or move not in game.possible_moves()
                or not game.move(game.pit_index(move))
            ):
                raise ValueError("seed429_suite_prefix_replay_failed")
        identity = suites.canonical_key(game.to_state())
        if identity != suites.canonical_key(row["state"]):
            raise ValueError("seed429_suite_state_identity_mismatch")
        suite_ids.add(identity)
    if len(suite_rows) != 512 or len(suite_ids) != 512 or suite_ids & final_ids:
        raise ValueError("seed429_suite_final_exclusion_check_failed")

    seed416 = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text(encoding="utf-8")
    )
    seed422 = json.loads(
        (ROOT / "docs/data/seed422-adam-first-moment/registration.json").read_text(
            encoding="utf-8"
        )
    )
    critical_historical_sources = {
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/arena.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/native_exact_root_tablebase.py",
        "ml/alphazero_lite/runtime_search_policy.py",
        "ml/alphazero_lite/export_artifact.py",
    }
    for relative, expected in seed416["source_hashes"].items():
        if relative not in critical_historical_sources:
            continue
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"seed429_historical_source_identity_mismatch:{relative}")
    for record in seed416["replays"]:
        derivative = Path(seed416["derivatives"][record["name"]]["A"]["derivative"])
        if (
            sha256_file(derivative)
            != seed416["derivatives"][record["name"]]["A"]["derivative_sha256"]
        ):
            raise ValueError(
                f"seed429_A_target_source_identity_mismatch:{record['name']}"
            )
    split_path = ROOT / seed416["training"]["source_row_split"]["path"]
    permutations_path = ROOT / seed416["training"]["epoch_permutations"]["path"]
    if sha256_file(split_path) != seed416["training"]["source_row_split"]["sha256"]:
        raise ValueError("seed429_frozen_split_identity_mismatch")
    if (
        sha256_file(permutations_path)
        != seed416["training"]["epoch_permutations"]["sha256"]
    ):
        raise ValueError("seed429_frozen_permutation_identity_mismatch")
    with gzip.open(permutations_path, "rt", encoding="utf-8") as stream:
        actual_permutations = json.load(stream)
    if {
        str(index + 1): hashlib.sha256(json.dumps(permutation).encode()).hexdigest()
        for index, permutation in enumerate(actual_permutations)
    } != seed416["training"]["epoch_permutations"]["epoch_sha256"]:
        raise ValueError("seed429_epoch_permutation_content_mismatch")
    init_path = Path(seed416["seed455_initialization"]["path"])
    if sha256_file(init_path) != seed416["seed455_initialization"]["sha256"]:
        raise ValueError("seed429_initializer_identity_mismatch")
    opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    for name, expected in seed422["evaluation"]["opponent_files"].items():
        if sha256_file(opponent / name) != expected:
            raise ValueError(f"seed429_frozen_opponent_identity_mismatch:{name}")
    runtime_items = {
        "native_probe": ROOT / "model-artifact/runtime/kalah_v1_tablebase",
        "tablebase": ROOT / "model-artifact/runtime/kalah_v1_21.kvtb",
        "runtime_policy": ROOT / "model-artifact/current/search_policy.json",
    }
    for name, path in runtime_items.items():
        if sha256_file(path) != seed422["evaluation"]["runtime_hashes"][name]:
            raise ValueError(f"seed429_runtime_identity_mismatch:{name}")
    source_inventory = {}
    for relative in EXECUTION_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"seed429_execution_source_missing:{relative}")
        source_inventory[relative] = sha256_file(path)
    reg_path = DATA / "registration.json"
    if require_registration:
        if not reg_path.is_file():
            raise ValueError("seed429_registration_missing")
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        if reg.get("status") != "registered_before_training_and_model_probing":
            raise ValueError("seed429_registration_status_invalid")
        if reg.get("census_sha256") != sha256_file(DATA / "coefficient-census.json"):
            raise ValueError("seed429_registration_census_binding_mismatch")
        if reg.get("suite_sha256") != sha256_file(suite_path):
            raise ValueError("seed429_registration_suite_binding_mismatch")
        if reg.get("exclusion_proof_sha256") != sha256_file(proof_path):
            raise ValueError("seed429_registration_exclusion_binding_mismatch")
        snapshots = reg.get("execution_source_snapshots", {})
        verify_execution_snapshots(ROOT, snapshots, EXECUTION_SOURCES)
    return {
        "ready": True,
        "no_training_or_games_started": True,
        "census_sha256": sha256_file(DATA / "coefficient-census.json"),
        "control_coefficients_sha256": census_record["control_coefficients_sha256"],
        "treatment_coefficients_sha256": census_record["treatment_coefficients_sha256"],
        "suite_sha256": sha256_file(suite_path),
        "exclusion_proof_sha256": sha256_file(proof_path),
        "excluded_identity_count": len(final_ids),
        "suite_count": len(suite_ids),
        "execution_source_inventory_count": len(source_inventory),
    }


def execution_readiness(*, require_candidates: bool = False) -> dict[str, Any]:
    """Validate registered bytes; optionally require completed training/runtime binding."""
    base = readiness(require_registration=True)
    if not require_candidates:
        return base
    registration_hash = sha256_file(DATA / "registration.json")
    training_path = DATA / "training-results.json"
    if not training_path.is_file():
        raise ValueError("seed429_completed_training_missing")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("registration_sha256") != registration_hash:
        raise ValueError("seed429_training_registration_binding_mismatch")
    for lane in ("A", "B"):
        lane_result = training["lanes"][lane]
        if lane_result["optimizer_updates"] != 1052:
            raise ValueError(f"seed429_training_updates_incomplete:{lane}")
        for epoch, expected in lane_result["epoch_hashes"].items():
            path = ROOT / lane_result["checkpoint_dir"] / f"E{epoch}.npz"
            if sha256_file(path) != expected:
                raise ValueError(
                    f"seed429_training_checkpoint_binding_mismatch:{lane}:E{epoch}"
                )
    if (
        training["lanes"]["A"]["epoch_hashes"]["4"]
        != "836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd"
    ):
        raise ValueError("seed429_A_E4_reproduction_missing")
    binding_path = DATA / "runtime-binding.json"
    if not binding_path.is_file():
        raise ValueError("seed429_candidate_runtime_binding_missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("registration_sha256") != registration_hash
        or binding.get("training_results_sha256") != sha256_file(training_path)
        or binding.get("suite_sha256") != sha256_file(DATA / "openings.jsonl")
    ):
        raise ValueError("seed429_candidate_runtime_binding_inconsistent")
    for lane in ("A", "B"):
        candidate = binding["candidates"][lane]
        artifact = ROOT / candidate["artifact"]
        try:
            verify_artifact_files(artifact, candidate["artifact_files"])
        except ValueError as error:
            raise ValueError(f"seed429_candidate_artifact_mismatch:{lane}") from error
    opponent_copy = ROOT / binding["opponent"]
    for filename, expected in binding["opponent_files"].items():
        if sha256_file(opponent_copy / filename) != expected:
            raise ValueError(f"seed429_published_opponent_artifact_mismatch:{filename}")
    local_opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    native_probe = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    for lane in ("A", "B"):
        local_artifact = (
            ROOT / ".tmp/seed429-canonical-policy-normalization/artifacts" / lane
        )
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=local_opponent,
            challenger_artifact=local_artifact,
            native_probe=native_probe,
            tablebase=tablebase,
        )
        if (
            contract is None
            or contract != binding["candidates"][lane]["runtime_contract"]
        ):
            raise ValueError(f"seed429_resolved_runtime_contract_mismatch:{lane}")
    return {**base, "training_complete": True, "candidate_runtime_bound": True}


def register() -> dict[str, Any]:
    """Freeze source snapshots and prospective protocol before any training."""
    preflight = readiness(require_registration=False)
    reg_path = DATA / "registration.json"
    if reg_path.exists():
        raise ValueError("seed429_registration_is_immutable")
    snapshot_dir = DATA / "execution-source-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_hashes = {}
    for relative in EXECUTION_SOURCES:
        source = ROOT / relative
        snapshot = snapshot_dir / Path(relative).name
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = source.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != payload:
            raise ValueError(f"seed429_snapshot_conflict:{relative}")
        if not snapshot.exists():
            snapshot.write_bytes(payload)
        snapshot_hashes[str(snapshot.relative_to(ROOT))] = sha256_file(snapshot)
    seed416 = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text(encoding="utf-8")
    )
    seed422 = json.loads(
        (ROOT / "docs/data/seed422-adam-first-moment/registration.json").read_text(
            encoding="utf-8"
        )
    )
    census_record = json.loads((DATA / "coefficient-census.json").read_text())
    init_path = Path(seed416["seed455_initialization"]["path"])
    portable_init = DATA / "training-checkpoints" / "initializer.npz"
    portable_init.parent.mkdir(parents=True, exist_ok=True)
    if not portable_init.exists():
        shutil.copy2(init_path, portable_init)
    if sha256_file(portable_init) != seed416["seed455_initialization"]["sha256"]:
        raise ValueError("seed429_portable_initializer_identity_mismatch")
    split_record = seed416["training"]["source_row_split"]
    permutation_record = seed416["training"]["epoch_permutations"]
    reg = {
        "schema": "seed429-canonical-policy-normalization-registration-v1",
        "status": "registered_before_training_and_model_probing",
        "hypothesis": "within-source canonical multiplicity normalization above 16 active stones improves strength",
        "arms": {
            "A": "exact #416 lane A four-epoch training",
            "B": "identical to A except compact training policy coefficients normalized by source and 17-32/>32 active-stone bucket",
        },
        "normalization": {
            "formula": "q'_r=q_r*Q/(K*Q_g) for positive Q_g; Q and Q_g are float64 sums of original float32 coefficients; converted once to float32 for production training",
            "inputs": "compact source rows before np.tile replay expansion",
            "validation_and_le_16_unchanged": True,
            "zero_coefficients_unchanged": True,
            "targets_and_rows_preserved": True,
            "per_batch_denominator": "production weighted_policy_loss sum(weighted row CE)/sum(batch coefficients)",
            "mass_note": "source/bucket compact coefficient mass conserved up to float32 conversion; realized minibatch gradient contributions need not be equal",
            "control_vector_sha256": census_record["control_coefficients_sha256"],
            "treatment_vector_sha256": census_record["treatment_coefficients_sha256"],
        },
        "source_registration": {
            "seed416_registration_v3_sha256": sha256_file(
                ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
            ),
            "replays": seed416["replays"],
            "A_policy_target_derivative_identities": {
                item["name"]: {
                    "path": seed416["derivatives"][item["name"]]["A"]["derivative"],
                    "sha256": seed416["derivatives"][item["name"]]["A"][
                        "derivative_sha256"
                    ],
                }
                for item in seed416["replays"]
            },
            "source_order_weights": [1, 4, 1, 8, 4],
            "split_path": split_record["path"],
            "split_sha256": split_record["sha256"],
            "permutation_path": permutation_record["path"],
            "permutation_sha256": permutation_record["sha256"],
        },
        "initializer": seed416["seed455_initialization"],
        "portable_initializer_path": str(portable_init.relative_to(ROOT)),
        "frozen_runtime": {
            "opponent_path": ".tmp/seed461-order-confirmation/opponent-artifact",
            "opponent_files": seed422["evaluation"]["opponent_files"],
            "runtime_hashes": seed422["evaluation"]["runtime_hashes"],
        },
        "suite_path": "docs/data/seed429-canonical-policy-normalization/openings.jsonl",
        "suite_sha256": sha256_file(DATA / "openings.jsonl"),
        "exclusion_proof_path": "docs/data/seed429-canonical-policy-normalization/opening-exclusion-proof.json",
        "exclusion_proof_sha256": sha256_file(DATA / "opening-exclusion-proof.json"),
        "census_sha256": sha256_file(DATA / "coefficient-census.json"),
        "training": {
            "seed": 416,
            "order_seed": 416,
            "model": "residual_v3",
            "hidden_sizes": [96, 3],
            "input_encoding": "kalah_v3",
            "epochs": 4,
            "batch_size": 512,
            "updates": 1052,
            "optimizer": "Adam",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": "none",
            "weight_decay": 0,
            "value_loss": "Huber",
            "huber_delta": 1,
            "value_loss_weight": 0.3,
            "gradient_clip": 1,
            "trainable_parameters": "all",
            "checkpoint": "fixed E4",
            "expected_A_epoch_hashes": json.loads(
                (
                    ROOT
                    / "docs/data/seed416-policy-target-softening/training-results.json"
                ).read_text()
            )["lanes"]["A"]["epochs"],
        },
        "evaluation": {
            "games": 2048,
            "games_per_arm": 1024,
            "both_seats": True,
            "opponent": "frozen seed455",
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "max_moves": 200,
            "resumable_chunk_games": 32,
            "worker_count_per_chunk": 1,
            "native_root_solve_threshold": 16,
            "search_options": arena_module.DEFAULT_EVAL_SEARCH_OPTIONS,
            "seed": 429,
            "seed_contract": "azlite_eval_seed_v2",
            "adaptive_extension": False,
            "bootstrap_samples": 20000,
            "bootstrap_seed": 429,
            "confidence": 0.95,
            "advance": "paired B-A >= 0.03 and paired lower bound > 0 and B score >= 0.53 and B lower bound > 0.50",
            "promotion": False,
        },
        "execution_source_snapshots": snapshot_hashes,
        "execution_source_inventory": list(EXECUTION_SOURCES),
        "commands": {
            "readiness": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization readiness",
            "registration": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization register",
            "training": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization train",
            "artifact_binding": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization bind",
            "arena": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization evaluate",
            "analysis": "python -m ml.alphazero_lite.run_seed429_canonical_policy_normalization analyze",
            "portable_verifier": "python -m ml.alphazero_lite.verify_seed429_publication",
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    if (
        reg["census_sha256"] != preflight["census_sha256"]
        or reg["suite_sha256"] != preflight["suite_sha256"]
        or reg["exclusion_proof_sha256"] != preflight["exclusion_proof_sha256"]
    ):
        raise ValueError("seed429_registration_input_binding_mismatch")
    if set(snapshot_hashes) != {
        str(
            (DATA / "execution-source-snapshots" / Path(relative).name).relative_to(
                ROOT
            )
        )
        for relative in EXECUTION_SOURCES
    }:
        raise ValueError("seed429_execution_source_inventory_incomplete")
    for relative, digest in snapshot_hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(
                f"seed429_execution_snapshot_precommit_mismatch:{relative}"
            )
    staging = reg_path.with_suffix(".json.staging")
    staging.write_text(
        json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, reg_path)
    execution_readiness()
    return reg


def bind_artifacts() -> dict[str, Any]:
    readiness(require_registration=True)
    reg_path = DATA / "registration.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    training_path = DATA / "training-results.json"
    if not training_path.is_file():
        raise ValueError("seed429_training_results_missing")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("registration_sha256") != sha256_file(reg_path):
        raise ValueError("seed429_training_registration_binding_mismatch")
    opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    native_probe = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    candidates = {}
    work = ROOT / ".tmp/seed429-canonical-policy-normalization"
    for lane in ("A", "B"):
        checkpoint = ROOT / training["lanes"][lane]["checkpoint_dir"] / "E4.npz"
        expected = training["lanes"][lane]["epoch_hashes"]["4"]
        if sha256_file(checkpoint) != expected:
            raise ValueError(f"seed429_E4_checkpoint_hash_mismatch:{lane}")
        artifact = work / "artifacts" / lane
        staging = work / "artifacts" / f"{lane}.staging"
        if not artifact.exists():
            staging.parent.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                raise ValueError(f"seed429_unexpected_artifact_staging:{lane}")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(staging),
                    "--version",
                    f"seed429-canonical-policy-{lane}-E4",
                    "--model-type",
                    "residual_v3",
                    "--rules-version",
                    "kalah_v1",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=ROOT,
                check=True,
            )
            shutil.copy2(
                ROOT / "model-artifact/current/search_policy.json",
                staging / "search_policy.json",
            )
            if sha256_file(staging / "model.npz") != expected:
                raise ValueError(f"seed429_exported_checkpoint_mismatch:{lane}")
            os.replace(staging, artifact)
        files = ("model.npz", "weights.json", "metadata.json", "search_policy.json")
        file_hashes = {name: sha256_file(artifact / name) for name in files}
        if file_hashes["model.npz"] != expected:
            raise ValueError(f"seed429_artifact_checkpoint_mismatch:{lane}")
        if (
            file_hashes["search_policy.json"]
            != reg["frozen_runtime"]["runtime_hashes"]["runtime_policy"]
        ):
            raise ValueError(f"seed429_artifact_sidecar_mismatch:{lane}")
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=opponent,
            challenger_artifact=artifact,
            native_probe=native_probe,
            tablebase=tablebase,
        )
        if contract is None:
            raise ValueError(f"seed429_runtime_contract_missing:{lane}")
        published_artifact = DATA / "runtime-artifacts" / lane
        published_artifact.mkdir(parents=True, exist_ok=True)
        for filename in files:
            destination = published_artifact / filename
            if not destination.exists():
                shutil.copy2(artifact / filename, destination)
            if sha256_file(destination) != file_hashes[filename]:
                raise ValueError(
                    f"seed429_published_runtime_artifact_mismatch:{lane}:{filename}"
                )
        candidates[lane] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": expected,
            "artifact": str(published_artifact.relative_to(ROOT)),
            "artifact_files": file_hashes,
            "runtime_contract": contract,
        }
    if candidates["A"]["runtime_contract"] != candidates["B"]["runtime_contract"]:
        raise ValueError("seed429_candidate_runtime_contract_mismatch")
    published_opponent = DATA / "runtime-artifacts" / "seed455"
    published_opponent.mkdir(parents=True, exist_ok=True)
    opponent_hashes = {}
    for filename in ("weights.json", "metadata.json", "search_policy.json"):
        destination = published_opponent / filename
        if not destination.exists():
            shutil.copy2(opponent / filename, destination)
        expected = reg["frozen_runtime"]["opponent_files"][filename]
        if sha256_file(destination) != expected:
            raise ValueError(f"seed429_published_opponent_mismatch:{filename}")
        opponent_hashes[filename] = expected
    binding = {
        "schema": "seed429-runtime-binding-v1",
        "registration_sha256": sha256_file(reg_path),
        "training_results_sha256": sha256_file(training_path),
        "suite_sha256": sha256_file(DATA / "openings.jsonl"),
        "opponent": str(published_opponent.relative_to(ROOT)),
        "opponent_files": opponent_hashes,
        "native_probe_sha256": sha256_file(native_probe),
        "tablebase_sha256": sha256_file(tablebase),
        "runtime_policy_sha256": sha256_file(
            ROOT / "model-artifact/current/search_policy.json"
        ),
        "runtime_contract": candidates["A"]["runtime_contract"],
        "candidates": candidates,
    }
    write_json(DATA / "runtime-binding.json", binding)
    return binding


def _validate_lane_arena(
    lane: str, report_path: Path, games_path: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    games = [
        json.loads(line)
        for line in games_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    suite_rows = suites.load_suite_jsonl(str(DATA / "openings.jsonl"))
    notes = report.get("notes", {})
    if len(games) != 1024 or report.get("games_played") != 1024:
        raise ValueError(f"seed429_lane_game_count_invalid:{lane}")
    if (
        notes.get("base_seed") != 429
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
    ):
        raise ValueError(f"seed429_lane_seed_contract_invalid:{lane}")
    if notes.get("suite_sha256") != sha256_file(DATA / "openings.jsonl"):
        raise ValueError(f"seed429_lane_suite_binding_invalid:{lane}")
    profile = notes.get("search_profile", {})
    if (
        profile.get("challenger_simulations") != 384
        or profile.get("current_simulations") != 384
        or profile.get("c_puct") != 1.25
    ):
        raise ValueError(f"seed429_lane_search_settings_invalid:{lane}")
    seen: set[tuple[int, int]] = set()
    contexts = set()
    for row in games:
        opening = int(row["opening_index"])
        seat = int(row["challenger_player"])
        if not 0 <= opening < 512 or seat not in (0, 1) or (opening, seat) in seen:
            raise ValueError(f"seed429_duplicate_or_invalid_game:{lane}")
        seen.add((opening, seat))
        if row.get("opening_contract") != "arena_player_relative_v2" or row.get(
            "opening_state_hash"
        ) != suite_rows[opening].get("state_hash"):
            raise ValueError(f"seed429_game_opening_binding_invalid:{lane}")
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(action) for action in suite_rows[opening]["prefix_moves"]]
        if apply_opening_moves(game, prefix) != len(prefix):
            raise ValueError(f"seed429_game_prefix_invalid:{lane}")
        trajectory = [
            int(action) for action in str(row["trajectory"]).split(",") if action
        ]
        for ply, action in enumerate(trajectory):
            if (
                game.over()
                or action < 0
                or action >= 12
                or action // 6 != game.current_player
                or not game.move(action)
            ):
                raise ValueError(
                    f"seed429_game_trajectory_invalid:{lane}:{opening}:{ply}"
                )
        if not game.over() or len(trajectory) != int(row["game_length"]):
            raise ValueError(f"seed429_game_incomplete:{lane}:{opening}")
        margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
        winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
        if margin != int(row["margin"]) or winner != row["winner"]:
            raise ValueError(
                f"seed429_game_terminal_accounting_invalid:{lane}:{opening}"
            )
    if len(seen) != 1024 or seen != {
        (opening, seat) for opening in range(512) for seat in (0, 1)
    }:
        raise ValueError(f"seed429_lane_seat_pairing_incomplete:{lane}")
    seed_ledger = Path(
        str(games_path).replace("-games.jsonl", "-seed-identities.jsonl")
    )
    if seed_ledger.is_file():
        with seed_ledger.open(encoding="utf-8") as stream:
            contexts = {
                row["seed_context_hash"]
                for line in stream
                if line.strip()
                for row in [json.loads(line)]
                if int(row.get("ply", -1)) == 0
            }
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "games_path": str(games_path),
        "games_sha256": sha256_file(games_path),
        "seed_contexts": sorted(contexts),
        "game_count": len(games),
    }


def _validate_chunk_rows(rows: list[dict[str, Any]], start: int, count: int) -> None:
    suite_rows = suites.load_suite_jsonl(str(DATA / "openings.jsonl"))
    if len(rows) != count:
        raise ValueError("seed429_chunk_game_count_invalid")
    seen = set()
    for row in rows:
        game_index = int(row["game_index"])
        opening = int(row["opening_index"])
        seat = int(row["challenger_player"])
        if not start <= game_index < start + count:
            raise ValueError("seed429_chunk_game_index_out_of_range")
        if game_index != opening * 2 + int(row["game_within_opening"]):
            raise ValueError("seed429_chunk_opening_index_mismatch")
        if (opening, seat) in seen or seat != game_index % 2:
            raise ValueError("seed429_chunk_duplicate_or_seat_mismatch")
        seen.add((opening, seat))
        if row.get("opening_contract") != "arena_player_relative_v2" or row.get(
            "opening_state_hash"
        ) != suite_rows[opening].get("state_hash"):
            raise ValueError("seed429_chunk_opening_identity_mismatch")
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in suite_rows[opening]["prefix_moves"]]
        if apply_opening_moves(game, prefix) != len(prefix):
            raise ValueError("seed429_chunk_prefix_replay_failed")
        trajectory = [
            int(action) for action in str(row["trajectory"]).split(",") if action
        ]
        for action in trajectory:
            if (
                game.over()
                or not 0 <= action < 12
                or action // 6 != game.current_player
                or not game.move(action)
            ):
                raise ValueError("seed429_chunk_trajectory_replay_failed")
        if not game.over() or len(trajectory) != int(row["game_length"]):
            raise ValueError("seed429_chunk_trajectory_incomplete")
        margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
        winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
        if margin != int(row["margin"]) or winner != row["winner"]:
            raise ValueError("seed429_chunk_terminal_accounting_invalid")


def _load_completed_chunk(
    chunk_path: Path,
    *,
    expected_sha256: str,
    lane: str,
    start: int,
    count: int,
    registration_sha256: str,
    runtime_binding_sha256: str,
    suite_sha256: str,
) -> dict[str, Any]:
    """Validate one persisted chunk before treating it as resumed arena work."""
    if not chunk_path.is_file() or sha256_file(chunk_path) != expected_sha256:
        raise ValueError("seed429_chunk_resume_hash_mismatch")
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    expected = {
        "lane": lane,
        "start_index": start,
        "game_count": count,
        "registration_sha256": registration_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "suite_sha256": suite_sha256,
        "evaluation_seed": 429,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
        "c_puct": 1.25,
    }
    for key, value in expected.items():
        if chunk.get(key) != value:
            raise ValueError(f"seed429_chunk_resume_identity_mismatch:{key}")
    _validate_chunk_rows(chunk["game_entries"], start, count)
    if len({int(row["game_index"]) for row in chunk["game_entries"]}) != count:
        raise ValueError("seed429_chunk_resume_duplicate_game")
    return chunk


def _run_arena_lane_resumable(
    lane: str, binding: dict[str, Any], *, chunk_size: int = 32
) -> tuple[Path, Path, Path, Path, Path]:
    """Execute fixed game-index chunks; commit each chunk before advancing."""
    registration_hash = sha256_file(DATA / "registration.json")
    binding_hash = sha256_file(DATA / "runtime-binding.json")
    suite_path = DATA / "openings.jsonl"
    work = ROOT / ".tmp/seed429-canonical-policy-normalization"
    lane_work = work / "arena-chunks" / lane
    lane_work.mkdir(parents=True, exist_ok=True)
    index_path = lane_work / "resume-index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("registration_sha256") != registration_hash
            or index.get("runtime_binding_sha256") != binding_hash
            or index.get("suite_sha256") != sha256_file(suite_path)
            or index.get("evaluation_seed") != 429
            or index.get("seed_contract") != "azlite_eval_seed_v2"
            or index.get("challenger_simulations") != 384
            or index.get("current_simulations") != 384
            or index.get("c_puct") != 1.25
        ):
            raise ValueError(f"seed429_chunk_resume_binding_mismatch:{lane}")
    else:
        index = {
            "registration_sha256": registration_hash,
            "runtime_binding_sha256": binding_hash,
            "suite_sha256": sha256_file(suite_path),
            "chunks": {},
        }
    candidate = ROOT / binding["candidates"][lane]["artifact"]
    opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    runtime_contract = resolve_strength_comparison_runtime_contract(
        current_artifact=opponent,
        challenger_artifact=ROOT
        / ".tmp/seed429-canonical-policy-normalization/artifacts"
        / lane,
        native_probe=ROOT / "model-artifact/runtime/kalah_v1_tablebase",
        tablebase=ROOT / "model-artifact/runtime/kalah_v1_21.kvtb",
    )
    if runtime_contract != binding["candidates"][lane]["runtime_contract"]:
        raise ValueError(f"seed429_chunk_runtime_contract_mismatch:{lane}")
    for start in range(0, 1024, chunk_size):
        count = min(chunk_size, 1024 - start)
        key = str(start)
        chunk_path = lane_work / f"chunk-{start:04d}.json"
        previous = index["chunks"].get(key)
        if previous is not None:
            _load_completed_chunk(
                chunk_path,
                expected_sha256=previous["sha256"],
                lane=lane,
                start=start,
                count=count,
                registration_sha256=registration_hash,
                runtime_binding_sha256=binding_hash,
                suite_sha256=sha256_file(suite_path),
            )
            continue
        if chunk_path.exists():
            # Recovery window: the chunk was atomically committed but the
            # separate progress index was not. Adopt it only after full binding
            # and strict replay; the execution is still one unique game-index set.
            orphan = json.loads(chunk_path.read_text(encoding="utf-8"))
            _load_completed_chunk(
                chunk_path,
                expected_sha256=sha256_file(chunk_path),
                lane=lane,
                start=start,
                count=count,
                registration_sha256=registration_hash,
                runtime_binding_sha256=binding_hash,
                suite_sha256=sha256_file(suite_path),
            )
            index["chunks"][key] = {
                "sha256": sha256_file(chunk_path),
                "game_count": int(orphan["game_count"]),
            }
            index.update(
                {
                    "evaluation_seed": 429,
                    "seed_contract": "azlite_eval_seed_v2",
                    "challenger_simulations": 384,
                    "current_simulations": 384,
                    "c_puct": 1.25,
                }
            )
            write_progress(index_path, index)
            continue
        staging_orphan = chunk_path.with_suffix(".json.staging")
        if staging_orphan.exists():
            staging_orphan.unlink()
        argv = [
            "--challenger",
            str(candidate),
            "--current",
            str(opponent),
            "--games",
            str(count),
            "--games-per-opening",
            "2",
            "--opening-prefixes-jsonl",
            str(suite_path),
            "--suite-sha256",
            sha256_file(suite_path),
            "--challenger-simulations",
            "384",
            "--current-simulations",
            "384",
            "--seed",
            "429",
            "--workers",
            "1",
            "--c-puct",
            "1.25",
            "--seed-contract",
            "azlite_eval_seed_v2",
        ]
        saved_argv = sys.argv
        try:
            sys.argv = ["arena.py", *argv]
            args = arena_module.parse_args()
        finally:
            sys.argv = saved_argv
        options = arena_module.build_eval_search_options(
            **arena_module.search_options_from_args(args)
        )
        challenger_options = arena_module.parse_search_options_override(
            args.challenger_search_options_json, base=options
        )
        current_options = arena_module.parse_search_options_override(
            args.current_search_options_json, base=options
        )
        worker = arena_module.run_arena_worker(
            worker_id=start // chunk_size,
            start_index=start,
            games=count,
            challenger_path=str(candidate),
            current_path=str(opponent),
            challenger_simulations=384,
            current_simulations=384,
            seed=429,
            c_puct=1.25,
            max_moves=args.max_moves,
            fpu_mode=str(options["fpu_mode"]),
            reuse_subtree=bool(options["reuse_subtree"]),
            normalize_values=bool(options["normalize_values"]),
            root_policy_mode=str(options["root_policy_mode"]),
            tactical_root_bias=float(options["tactical_root_bias"]),
            root_temperature=float(options.get("root_temperature", 0.0)),
            challenger_search_options=challenger_options,
            current_search_options=current_options,
            exact_root_solve_threshold=runtime_contract["exact_root_solve_threshold"],
            exact_root_native_probe=Path(runtime_contract["exact_root_native_probe"]),
            exact_root_tablebase_path=Path(runtime_contract["exact_root_tablebase"]),
            runtime_search_contract=runtime_contract,
            opening_prefixes_jsonl=str(suite_path),
            games_per_opening=2,
            seed_contract="azlite_eval_seed_v2",
            suite_sha256_override=sha256_file(suite_path),
        )
        game_entries = worker["game_entries"]
        _validate_chunk_rows(game_entries, start, count)
        chunk = {
            "lane": lane,
            "start_index": start,
            "game_count": count,
            "registration_sha256": registration_hash,
            "runtime_binding_sha256": binding_hash,
            "suite_sha256": sha256_file(suite_path),
            "evaluation_seed": 429,
            "seed_contract": "azlite_eval_seed_v2",
            "challenger_simulations": 384,
            "current_simulations": 384,
            "c_puct": 1.25,
            "search_profile": worker["search_profile"],
            "game_entries": game_entries,
            "seed_identity_ledger": worker["seed_identity_ledger"],
            "search_configuration_ledger": worker["search_configuration_ledger"],
            "search_outcome_ledger": worker["search_outcome_ledger"],
        }
        staging = chunk_path.with_suffix(".json.staging")
        staging.write_text(
            json.dumps(chunk, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        os.replace(staging, chunk_path)
        index["chunks"][key] = {"sha256": sha256_file(chunk_path), "game_count": count}
        index.update(
            {
                "evaluation_seed": 429,
                "seed_contract": "azlite_eval_seed_v2",
                "challenger_simulations": 384,
                "current_simulations": 384,
                "c_puct": 1.25,
            }
        )
        write_progress(index_path, index)
    chunks = [
        json.loads((lane_work / f"chunk-{start:04d}.json").read_text(encoding="utf-8"))
        for start in range(0, 1024, chunk_size)
    ]
    games = [row for chunk in chunks for row in chunk["game_entries"]]
    if len(games) != 1024 or len({int(row["game_index"]) for row in games}) != 1024:
        raise ValueError(f"seed429_lane_chunks_incomplete:{lane}")
    games_path = work / f"{lane}-games.jsonl"
    seed_path = work / f"{lane}-seed-identities.jsonl"
    config_path = work / f"{lane}-search-configurations.jsonl"
    search_path = work / f"{lane}-search-outcomes.jsonl"
    games_path.write_text(
        "".join(json.dumps(row) + "\n" for row in games), encoding="utf-8"
    )
    for filename, field in (
        (seed_path, "seed_identity_ledger"),
        (config_path, "search_configuration_ledger"),
        (search_path, "search_outcome_ledger"),
    ):
        entries = [row for chunk in chunks for row in chunk[field]]
        filename.write_text(
            "".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8"
        )
    wins = sum(row["winner"] == "challenger" for row in games)
    losses = sum(row["winner"] == "current" for row in games)
    draws = sum(row["winner"] == "draw" for row in games)
    report = {
        "games_played": 1024,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / 1024,
        "notes": {
            "base_seed": 429,
            "seed_contract": "azlite_eval_seed_v2",
            "suite_sha256": sha256_file(suite_path),
            "search_profile": chunks[0]["search_profile"],
        },
    }
    report_path = work / f"{lane}-arena.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, games_path, seed_path, config_path, search_path


def evaluate() -> dict[str, Any]:
    execution_readiness(require_candidates=True)
    binding_path = DATA / "runtime-binding.json"
    if not binding_path.is_file():
        raise ValueError("seed429_runtime_binding_missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["registration_sha256"] != sha256_file(DATA / "registration.json"):
        raise ValueError("seed429_runtime_binding_registration_mismatch")
    outcomes_path = DATA / "outcome-binding.json"
    previous = (
        json.loads(outcomes_path.read_text())
        if outcomes_path.exists()
        else {
            "registration_sha256": binding["registration_sha256"],
            "runtime_binding_sha256": sha256_file(binding_path),
            "lanes": {},
        }
    )
    if previous.get("registration_sha256") != binding[
        "registration_sha256"
    ] or previous.get("runtime_binding_sha256") != sha256_file(binding_path):
        raise ValueError("seed429_arena_resume_binding_mismatch")
    lanes = dict(previous.get("lanes", {}))
    for lane in ("A", "B"):
        old = lanes.get(lane)
        if old is not None:
            for path_key, digest_key in (
                ("games_path", "games_sha256"),
                ("report_path", "report_sha256"),
                ("seed_identity_ledger_path", "seed_identity_ledger_sha256"),
                (
                    "search_configuration_ledger_path",
                    "search_configuration_ledger_sha256",
                ),
                ("search_outcome_ledger_path", "search_outcome_ledger_sha256"),
            ):
                previous_path = Path(old[path_key])
                if not previous_path.is_absolute():
                    previous_path = ROOT / previous_path
                if sha256_file(previous_path) != old[digest_key]:
                    raise ValueError(
                        f"seed429_completed_outcome_altered:{lane}:{path_key}"
                    )
            report = ROOT / old["report_path"]
            games = ROOT / old["games_path"]
            _validate_lane_arena(lane, report, games)
            validated = old
            seed_ledger = ROOT / old["seed_identity_ledger_path"]
            config_ledger = ROOT / old["search_configuration_ledger_path"]
            search_ledger = ROOT / old["search_outcome_ledger_path"]
        else:
            report, games, seed_ledger, config_ledger, search_ledger = (
                _run_arena_lane_resumable(lane, binding)
            )
            validated = _validate_lane_arena(lane, report, games)
        for kind, path in (
            ("seed_identity_ledger", seed_ledger),
            ("search_configuration_ledger", config_ledger),
            ("search_outcome_ledger", search_ledger),
        ):
            if not path.is_file():
                raise ValueError(f"seed429_arena_provenance_missing:{lane}:{kind}")
            if old is None:
                validated[f"{kind}_sha256"] = sha256_file(path)
                validated[f"{kind}_path"] = str(path)
        if old is None:
            lane_publication = DATA / "arena-evidence" / lane
            lane_publication.mkdir(parents=True, exist_ok=True)
            published_files = {
                "report_path": report,
                "games_path": games,
                "seed_identity_ledger_path": seed_ledger,
                "search_configuration_ledger_path": config_ledger,
                "search_outcome_ledger_path": search_ledger,
            }
            for field, source in published_files.items():
                destination = lane_publication / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
                if sha256_file(destination) != sha256_file(source):
                    raise ValueError(
                        f"seed429_arena_publication_copy_mismatch:{lane}:{field}"
                    )
                validated[field] = str(destination.relative_to(ROOT))
        if old is not None and old != validated:
            raise ValueError(f"seed429_arena_resume_outcome_conflict:{lane}")
        lanes[lane] = validated
        write_progress(
            outcomes_path,
            {
                "schema": "seed429-outcome-binding-v1",
                "status": "in_progress",
                "registration_sha256": binding["registration_sha256"],
                "runtime_binding_sha256": sha256_file(binding_path),
                "lanes": lanes,
            },
        )
    if len(lanes) != 2 or any(lanes[lane]["game_count"] != 1024 for lane in ("A", "B")):
        raise ValueError("seed429_arena_final_evidence_incomplete")
    if set(lanes["A"]["seed_contexts"]) != set(lanes["B"]["seed_contexts"]):
        raise ValueError("seed429_paired_seed_context_mismatch")
    result = {
        "schema": "seed429-outcome-binding-v1",
        "status": "completed_2048_games",
        "registration_sha256": binding["registration_sha256"],
        "runtime_binding_sha256": sha256_file(binding_path),
        "lanes": lanes,
    }
    write_progress(outcomes_path, result)
    return result


def analyze() -> dict[str, Any]:
    """Publish the fixed paired-opening bootstrap and no-promotion decision."""
    from ml.alphazero_lite.seed416_policy_target_softening import bootstrap_paired

    outcomes_path = DATA / "outcome-binding.json"
    if not outcomes_path.is_file():
        raise ValueError("seed429_incomplete_arena_cannot_be_analyzed")
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    if outcomes.get("status") != "completed_2048_games":
        raise ValueError("seed429_incomplete_arena_cannot_be_analyzed")
    if outcomes["registration_sha256"] != sha256_file(DATA / "registration.json"):
        raise ValueError("seed429_analysis_registration_mismatch")
    rows: list[dict[str, Any]] = []
    lanes = {}
    for lane in ("A", "B"):
        evidence = outcomes["lanes"][lane]
        game_path = ROOT / evidence["games_path"]
        report_path = ROOT / evidence["report_path"]
        if (
            sha256_file(game_path) != evidence["games_sha256"]
            or sha256_file(report_path) != evidence["report_sha256"]
        ):
            raise ValueError(f"seed429_analysis_outcome_hash_mismatch:{lane}")
        games = [
            json.loads(line) for line in game_path.read_text().splitlines() if line
        ]
        if len(games) != 1024:
            raise ValueError(f"seed429_analysis_game_count_invalid:{lane}")
        lanes[lane] = games
        for game in games:
            winner = game["winner"]
            score = (
                1.0
                if winner == "challenger"
                else 0.5
                if winner == "draw"
                else 0.0
                if winner == "current"
                else -1.0
            )
            if score < 0:
                raise ValueError("seed429_analysis_invalid_winner")
            rows.append(
                {
                    "lane": lane,
                    "opening_id": str(int(game["opening_index"])),
                    "opponent_score": score,
                    "game": game,
                }
            )
    # The fixed subset check in bootstrap_paired also requires all 512 matched clusters.
    analysis = bootstrap_paired(rows, samples=20_000, seed=429)
    matrix = analysis["per_opening"]
    if len(matrix) != 512:
        raise ValueError("seed429_opening_matrix_incomplete")
    ledger_payload = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
    )
    ledger_path = DATA / "outcome-ledger.jsonl"
    if (
        ledger_path.exists()
        and ledger_path.read_text(encoding="utf-8") != ledger_payload
    ):
        raise ValueError("seed429_published_ledger_conflict")
    if not ledger_path.exists():
        ledger_path.write_text(ledger_payload, encoding="utf-8")
    matrix_path = DATA / "per-opening-matrix.json"
    write_json(matrix_path, matrix)
    analysis["outcome_ledger_sha256"] = sha256_file(ledger_path)
    analysis["registration_sha256"] = sha256_file(DATA / "registration.json")
    analysis["outcome_binding_sha256"] = sha256_file(outcomes_path)
    analysis["decision"] = (
        "advance_to_independent_replication"
        if analysis["decision"] == "advance_to_separate_confirmation"
        else "stop_normalization_branch"
    )
    write_json(DATA / "analysis.json", analysis)
    report = (
        "# Seed429 within-source canonical-multiplicity normalization\n\n"
        "The arena decision is fixed by the prospective registration. Passing only "
        "warrants independent replication; this result does not promote a checkpoint.\n\n"
        f"- Games: 2,048; openings: 512; both seats.\n"
        f"- Paired B−A: {analysis['primary_mean_B_minus_A']:.6f}; 95% interval "
        f"[{analysis['primary_95_percentile_interval'][0]:.6f}, "
        f"{analysis['primary_95_percentile_interval'][1]:.6f}].\n"
        f"- B score: {analysis['B_opponent_score_mean']:.6f}; 95% interval "
        f"[{analysis['B_opponent_score_95_percentile_interval'][0]:.6f}, "
        f"{analysis['B_opponent_score_95_percentile_interval'][1]:.6f}].\n"
        f"- Decision: **{analysis['decision']}**.\n"
    )
    report_path = DATA / "results.md"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise ValueError("seed429_results_report_conflict")
    if not report_path.exists():
        report_path.write_text(report, encoding="utf-8")
    return analysis


def retrospective_diagnostics() -> dict[str, Any]:
    """Evaluate initializer/A/B under unchanged #427 membership and coefficients."""
    import gzip
    from ml.alphazero_lite.seed427_validation_metrics import aggregate, row_losses

    training = json.loads((DATA / "training-results.json").read_text())
    checkpoint_paths = {
        "initializer": ROOT
        / json.loads((DATA / "registration.json").read_text())[
            "portable_initializer_path"
        ],
        "A_E4": ROOT / training["lanes"]["A"]["checkpoint_dir"] / "E4.npz",
        "B_E4": ROOT / training["lanes"]["B"]["checkpoint_dir"] / "E4.npz",
    }
    evidence_path = (
        ROOT
        / "docs/data/seed426-canonical-overlap/seed427-prediction-evidence.jsonl.gz"
    )
    membership_path = (
        ROOT
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(evidence_path, "rt", encoding="utf-8") as stream:
        base_rows = [json.loads(line) for line in stream if line.strip()]
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        membership_rows = [json.loads(line) for line in stream if line.strip()]
    membership_by_position = {
        int(row["weighted_position"]): row for row in membership_rows
    }
    if len(membership_by_position) != 14946:
        raise ValueError("seed429_validation_membership_count_mismatch")
    for row in base_rows:
        member = row["membership"]
        expected_member = membership_by_position.get(int(member["weighted_position"]))
        if expected_member != member:
            raise ValueError("seed429_seed427_membership_binding_mismatch")
    result: dict[str, Any] = {}
    diagnostic_evidence = []
    for model_name, checkpoint in checkpoint_paths.items():
        if model_name in ("initializer", "A_E4"):
            selected = [
                row
                for row in base_rows
                if row["model"]
                == ("initializer" if model_name == "initializer" else "e4")
            ]
            rows = []
            for row in selected:
                membership = row["membership"]
                losses = row["losses"]
                diagnostic_row = {
                    "model": model_name,
                    "membership": membership,
                    "losses": losses,
                }
                rows.append(diagnostic_row)
                diagnostic_evidence.append(diagnostic_row)
            result[model_name] = aggregate(rows)
            continue
        model = PolicyValueNet((96, 3), "residual_v3", 27)
        load_checkpoint_into_model(model, checkpoint)
        model.eval()
        rows = []
        with torch.no_grad():
            for source in base_rows:
                if source["model"] != "initializer":
                    continue
                membership = source["membership"]
                encoded = np.frombuffer(
                    bytes.fromhex(membership["input_identity"]), dtype="<f4"
                ).copy()
                logits, value_prediction = model(
                    torch.from_numpy(encoded).reshape(1, -1)
                )
                losses = row_losses(
                    logits[0].cpu().tolist(),
                    float(value_prediction.item()),
                    source["target_policy"],
                    float(source["target_value"]),
                    source["legal_mask"],
                    float(source["policy_weight"]),
                )
                diagnostic_row = {
                    "model": model_name,
                    "membership": membership,
                    "losses": losses,
                }
                rows.append(diagnostic_row)
                diagnostic_evidence.append(diagnostic_row)
        result[model_name] = aggregate(rows)
    evidence_path_out = DATA / "retrospective-diagnostic-evidence.jsonl.gz"
    evidence_bytes = gzip.compress(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in diagnostic_evidence
        ).encode("utf-8"),
        mtime=0,
    )
    if evidence_path_out.exists() and evidence_path_out.read_bytes() != evidence_bytes:
        raise ValueError("seed429_diagnostic_evidence_conflict")
    if not evidence_path_out.exists():
        evidence_path_out.write_bytes(evidence_bytes)
    binding = {
        "schema": "seed429-retrospective-diagnostics-v1",
        "classification": "retrospective diagnostics; not checkpoint selection or arena gate",
        "membership_sha256": sha256_file(membership_path),
        "prediction_evidence_sha256": sha256_file(evidence_path),
        "diagnostic_evidence_sha256": sha256_file(evidence_path_out),
        "checkpoints": {
            name: sha256_file(path) for name, path in checkpoint_paths.items()
        },
        "aggregates": result,
    }
    write_json(DATA / "retrospective-diagnostics.json", binding)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "census",
            "suite",
            "readiness",
            "pre-registration-readiness",
            "execution-readiness",
            "arena-readiness",
            "register",
            "train",
            "bind",
            "evaluate",
            "analyze",
        ),
    )
    args = parser.parse_args()
    if args.stage == "census":
        result = census()
        print(
            json.dumps(
                {
                    "classification": result["classification"],
                    "rows_changed": result["rows_changed"],
                    "control_coefficients_sha256": result[
                        "control_coefficients_sha256"
                    ],
                    "treatment_coefficients_sha256": result[
                        "treatment_coefficients_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
    elif args.stage == "suite":
        suite_path, proof = freeze_suite()
        print(
            json.dumps(
                {
                    "suite": str(suite_path.relative_to(ROOT)),
                    "suite_sha256": sha256_file(suite_path),
                    "exclusion_count": proof["combined_identity_count"],
                    "exclusion_identity_set_sha256": proof[
                        "combined_identity_set_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
    elif args.stage == "train":
        print(json.dumps(train(), indent=2, sort_keys=True))
    elif args.stage == "readiness":
        print(json.dumps(readiness(), indent=2, sort_keys=True))
    elif args.stage == "pre-registration-readiness":
        print(
            json.dumps(readiness(require_registration=False), indent=2, sort_keys=True)
        )
    elif args.stage == "execution-readiness":
        print(json.dumps(execution_readiness(), indent=2, sort_keys=True))
    elif args.stage == "arena-readiness":
        print(
            json.dumps(
                execution_readiness(require_candidates=True), indent=2, sort_keys=True
            )
        )
    elif args.stage == "register":
        print(json.dumps(register(), indent=2, sort_keys=True))
    elif args.stage == "bind":
        print(json.dumps(bind_artifacts(), indent=2, sort_keys=True))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate(), indent=2, sort_keys=True))
    elif args.stage == "analyze":
        print(json.dumps(analyze(), indent=2, sort_keys=True))
    elif args.stage == "diagnostics":
        print(json.dumps(retrospective_diagnostics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
