"""Read-only verifier for a relocated seed429 publication."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed429_policy_normalization import (
    sha256_file,
)
from ml.alphazero_lite.seed416_policy_target_softening import bootstrap_paired
from ml.alphazero_lite.seed427_validation_metrics import aggregate

EXECUTION_SOURCE_INVENTORY = (
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


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _independent_coefficients(root: Path) -> dict[str, Any]:
    """Rebuild compact vectors from portable compressed source snapshots."""
    historic_root = root / "docs/data/seed416-policy-target-softening"
    seed416 = json.loads((historic_root / "registration-v3.json").read_text())
    rows_by_source: list[tuple[str, int, list[dict[str, Any]]]] = []
    replay_map: list[int] = []
    identities: list[str] = []
    stones: list[int] = []
    q: list[float] = []
    source_names: list[str] = []
    source_records = seed416["replays"]
    sources_root = root / "docs/data/seed426-canonical-overlap/sources"
    for source in source_records:
        name = source["name"]
        snapshot = sources_root / f"{name}.jsonl.gz"
        raw = gzip.decompress(snapshot.read_bytes())
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"portable_replay_hash_mismatch:{name}")
        parsed = [json.loads(line) for line in raw.splitlines() if line]
        rows_by_source.append((name, int(source["weight"]), parsed))
        start = len(identities)
        for row in parsed:
            encoded = np.asarray(row["state"], dtype=np.float32)
            if encoded.shape != (27,) or not np.isfinite(encoded).all():
                raise ValueError(f"portable_replay_state_invalid:{name}")
            state = {
                "player_pits": np.rint(encoded[:6] * 48).astype(int).tolist(),
                "opponent_pits": np.rint(encoded[6:12] * 48).astype(int).tolist(),
                "player_store": int(round(float(encoded[12]) * 48)),
                "opponent_store": int(round(float(encoded[13]) * 48)),
                "current_player": int(round(float(encoded[14]))),
            }
            identities.append(suites.canonical_key(state))
            stones.append(sum(state["player_pits"]) + sum(state["opponent_pits"]))
            teacher = row.get("teacher_source") == "exact_root_tablebase"
            coefficient = 1.0
            if teacher:
                if (
                    int(str(row.get("active_pit_stones", -1))) > 16
                    or int(str(row.get("puct_simulations_executed", -1))) != 0
                ):
                    raise ValueError("portable_exact_root_coefficient_metadata_invalid")
            q.append(coefficient)
            source_names.append(name)
        end = len(identities)
        replay_map.extend(
            np.tile(
                np.arange(start, end, dtype=np.int64), int(source["weight"])
            ).tolist()
        )
    split_path = root / seed416["training"]["source_row_split"]["path"]
    if (
        hashlib.sha256(split_path.read_bytes()).hexdigest()
        != seed416["training"]["source_row_split"]["sha256"]
    ):
        raise ValueError("portable_frozen_split_hash_mismatch")
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    valid_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    if (
        hashlib.sha256(train_positions.tobytes()).hexdigest()
        != seed416["training"]["source_row_split"]["train_positions_sha256"]
    ):
        raise ValueError("portable_train_positions_hash_mismatch")
    if (
        hashlib.sha256(valid_positions.tobytes()).hexdigest()
        != seed416["training"]["source_row_split"]["validation_positions_sha256"]
    ):
        raise ValueError("portable_validation_positions_hash_mismatch")
    if set(train_positions) & set(valid_positions) or len(train_positions) + len(
        valid_positions
    ) != len(replay_map):
        raise ValueError("portable_split_partition_invalid")
    train_rows = set(replay_map[index] for index in train_positions)
    valid_rows = set(replay_map[index] for index in valid_positions)
    if train_rows & valid_rows or train_rows | valid_rows != set(range(len(q))):
        raise ValueError("portable_compact_split_mapping_invalid")
    control = np.asarray(q, dtype=np.float32)
    treatment = control.copy()
    group_rows: dict[tuple[str, str, str], list[int]] = {}
    for index in train_rows:
        if stones[index] <= 16:
            continue
        bucket = "17-32" if stones[index] <= 32 else ">32"
        group_rows.setdefault(
            (source_names[index], bucket, identities[index]), []
        ).append(index)
    scopes = {(source, bucket) for source, bucket, _ in group_rows}
    for source, bucket in scopes:
        identities_in_scope = {
            identity: members
            for (row_source, row_bucket, identity), members in group_rows.items()
            if row_source == source and row_bucket == bucket
        }
        unique_rows = sorted(
            {index for members in identities_in_scope.values() for index in members}
        )
        positive_groups = len(identities_in_scope)
        total = float(control[unique_rows].astype(np.float64).sum())
        for identity, members in identities_in_scope.items():
            group_mass = float(control[members].astype(np.float64).sum())
            factor = total / (positive_groups * group_mass)
            treatment[members] = np.asarray(
                control[members].astype(np.float64) * factor, dtype=np.float32
            )
    return {
        "control_coefficients_sha256": hashlib.sha256(
            control.astype("<f4").tobytes()
        ).hexdigest(),
        "treatment_coefficients_sha256": hashlib.sha256(
            treatment.astype("<f4").tobytes()
        ).hexdigest(),
        "rows_changed": int(np.count_nonzero(control != treatment)),
        "source_row_count": len(control),
        "expanded_replay_position_count": len(replay_map),
        "training_compact_row_count": len(train_rows),
        "validation_compact_row_count": len(valid_rows),
        "validation_unchanged": bool(
            np.array_equal(control[list(valid_rows)], treatment[list(valid_rows)])
        ),
        "le16_unchanged": bool(
            np.array_equal(
                control[np.asarray(stones) <= 16], treatment[np.asarray(stones) <= 16]
            )
        ),
    }


def _verify_components(root: Path, proof: dict[str, Any]) -> None:
    excluded = set(proof["excluded_identities"])
    if len(excluded) != int(proof["combined_identity_count"]):
        raise ValueError("exclusion_identity_count_mismatch")
    if (
        hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
        != proof["combined_identity_set_sha256"]
    ):
        raise ValueError("exclusion_identity_hash_mismatch")
    for component in proof["components"]:
        path = root / component["path"]
        if not path.is_file() or sha256_file(path) != component["sha256"]:
            raise ValueError(f"exclusion_component_hash_mismatch:{component['name']}")
    # Reconstruct the canonical states in the #426/#427 evidence using its
    # registered JSON tuple encoding, then hash through the common Kalah key.
    for name in (
        "seed426-row-accounting",
        "seed427-validation-membership",
        "seed427-model-evaluated-predictions",
    ):
        component = next(item for item in proof["components"] if item["name"] == name)
        identities: set[str] = set()
        with gzip.open(root / component["path"], "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                member = row.get("membership", row)
                fields = json.loads(member["canonical_identity"])
                state = {
                    "player_pits": fields[0],
                    "opponent_pits": fields[1],
                    "player_store": fields[2],
                    "opponent_store": fields[3],
                    "current_player": fields[4],
                }
                identities.add(suites.canonical_key(state))
        if len(identities) != component["declared_or_evaluated_identity_count"]:
            raise ValueError(f"retrospective_identity_count_mismatch:{name}")
        if not identities <= excluded:
            raise ValueError(f"retrospective_identity_missing_from_union:{name}")
        if component["identities_added_to_union"] != 0:
            raise ValueError(f"unexpected_retrospective_exclusion_addition:{name}")
    _rebuild_final_union(root, proof)


def _rebuild_final_union(root: Path, proof: dict[str, Any]) -> None:
    """Reconcile frozen historical IDs, #422 trajectories, and #426–428 states."""
    from ml.alphazero_lite.seed422_exclusions import replay_game, replay_opening

    historical_path = (
        root / "docs/data/seed422-adam-first-moment/opening-exclusion-proof.json"
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_ids = set(historical["excluded_identities"])
    if len(historical_ids) != int(historical["identity_count"]):
        raise ValueError("historical_exclusion_count_mismatch")
    historical_digest = hashlib.sha256(
        "\n".join(sorted(historical_ids)).encode()
    ).hexdigest()
    if historical_digest != historical["identity_set_sha256"]:
        raise ValueError("historical_exclusion_digest_mismatch")
    if (
        historical_digest
        != proof["historical_union_through_421"]["identity_set_sha256"]
    ):
        raise ValueError("historical_exclusion_component_binding_mismatch")
    suite_path = root / "docs/data/seed422-adam-first-moment/openings.jsonl"
    ledger_path = root / "docs/data/seed422-adam-first-moment/outcome-ledger.jsonl"
    suite_rows = _rows(suite_path)
    outcome_rows = _rows(ledger_path)
    if len(suite_rows) != 512 or len(outcome_rows) != 2048:
        raise ValueError("seed422_exclusion_evidence_incomplete")
    roots = set()
    for opening in suite_rows:
        game = replay_opening(opening)
        roots.add(suites.canonical_key(game.to_state()))
    consumed = set()
    outcomes = set()
    for row in outcome_rows:
        game = row["game"]
        opening = int(row["opening_id"])
        key = (row["lane"], opening, int(game["challenger_player"]))
        if key in outcomes or not 0 <= opening < 512:
            raise ValueError("seed422_exclusion_duplicate_outcome")
        outcomes.add(key)
        consumed.update(replay_game(suite_rows[opening], game))
    expected_outcomes = {
        (lane, opening, seat)
        for lane in ("A", "B")
        for opening in range(512)
        for seat in (0, 1)
    }
    if outcomes != expected_outcomes:
        raise ValueError("seed422_exclusion_outcome_coverage_mismatch")
    union = historical_ids | roots | consumed
    for name in (
        "seed426-row-accounting",
        "seed427-validation-membership",
        "seed427-model-evaluated-predictions",
    ):
        component = next(row for row in proof["components"] if row["name"] == name)
        identities = set()
        with gzip.open(root / component["path"], "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                member = row.get("membership", row)
                fields = json.loads(member["canonical_identity"])
                identities.add(
                    suites.canonical_key(
                        {
                            "player_pits": fields[0],
                            "opponent_pits": fields[1],
                            "player_store": fields[2],
                            "opponent_store": fields[3],
                            "current_player": fields[4],
                        }
                    )
                )
        if component["identities_added_to_union"] != len(identities - union):
            raise ValueError(f"retrospective_addition_count_mismatch:{name}")
        union.update(identities)
    if union != set(proof["excluded_identities"]):
        raise ValueError("final_exclusion_union_reconstruction_mismatch")


def _verify_suite(root: Path, reg: dict[str, Any], excluded: set[str]) -> None:
    path = root / reg["suite_path"]
    if sha256_file(path) != reg["suite_sha256"]:
        raise ValueError("suite_hash_mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids: set[str] = set()
    if len(rows) != 512:
        raise ValueError("suite_count_mismatch")
    for row in rows:
        game = KalahGame.from_state(suites.INITIAL_STATE)
        for move in row["prefix_moves"]:
            move = int(move)
            if (
                game.over()
                or move not in game.possible_moves()
                or not game.move(game.pit_index(move))
            ):
                raise ValueError("suite_prefix_replay_failed")
        identity = suites.canonical_key(game.to_state())
        if identity != suites.canonical_key(row["state"]):
            raise ValueError("suite_state_identity_mismatch")
        if identity in ids or identity in excluded:
            raise ValueError("suite_overlap_or_duplicate")
        if game.over() or int(row["pit_sum"]) <= 32:
            raise ValueError("suite_opening_ineligible")
        ids.add(identity)
    if len(ids) != 512:
        raise ValueError("suite_unique_count_mismatch")


def verify(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Verify bindings and, by default, complete published experiment evidence."""
    root = root.resolve()
    data = root / "docs/data/seed429-canonical-policy-normalization"
    reg_path = data / "registration.json"
    if not reg_path.is_file():
        raise ValueError("registration_missing")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    if reg.get("status") != "registered_before_training_and_model_probing":
        raise ValueError("registration_status_invalid")
    if tuple(reg.get("execution_source_inventory", ())) != EXECUTION_SOURCE_INVENTORY:
        raise ValueError("execution_source_inventory_incomplete")
    expected_snapshot_paths = {
        f"docs/data/seed429-canonical-policy-normalization/execution-source-snapshots/{Path(name).name}"
        for name in EXECUTION_SOURCE_INVENTORY
    }
    if set(reg.get("execution_source_snapshots", {})) != expected_snapshot_paths:
        raise ValueError("execution_snapshot_bindings_incomplete")
    for relative, digest in reg["execution_source_snapshots"].items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"execution_snapshot_mismatch:{relative}")
    census_path = data / "coefficient-census.json"
    if sha256_file(census_path) != reg["census_sha256"]:
        raise ValueError("census_binding_mismatch")
    reconstructed = _independent_coefficients(root)
    published_census = json.loads(census_path.read_text(encoding="utf-8"))
    for key in (
        "control_coefficients_sha256",
        "treatment_coefficients_sha256",
        "rows_changed",
        "validation_coefficients_unchanged",
        "low_stone_coefficients_unchanged",
    ):
        actual_key = {
            "validation_coefficients_unchanged": "validation_unchanged",
            "low_stone_coefficients_unchanged": "le16_unchanged",
        }.get(key, key)
        if reconstructed[actual_key] != published_census[key]:
            raise ValueError(f"coefficient_reconstruction_mismatch:{key}")
    proof_path = data / reg["exclusion_proof_path"].split("/")[-1]
    if sha256_file(proof_path) != reg["exclusion_proof_sha256"]:
        raise ValueError("exclusion_proof_binding_mismatch")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    _verify_components(root, proof)
    _verify_suite(root, reg, set(proof["excluded_identities"]))
    if not require_complete:
        return {"valid": True, "complete": False, "repository_root": str(root)}

    training_path = data / "training-results.json"
    runtime_path = data / "runtime-binding.json"
    outcomes_path = data / "outcome-binding.json"
    analysis_path = data / "analysis.json"
    diagnostics_path = data / "retrospective-diagnostics.json"
    diagnostic_evidence_path = data / "retrospective-diagnostic-evidence.jsonl.gz"
    required = (
        training_path,
        runtime_path,
        outcomes_path,
        analysis_path,
        diagnostics_path,
        diagnostic_evidence_path,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError("publication_incomplete:" + ",".join(missing))
    if not _verify_training(root, reg, training_path, runtime_path):
        raise ValueError("training_or_runtime_binding_invalid")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    ledger_path = data / "outcome-ledger.jsonl"
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != analysis["outcome_ledger_sha256"]
    ):
        raise ValueError("outcome_ledger_binding_invalid")
    rows = _rows(ledger_path)
    if len(rows) != 2048:
        raise ValueError("outcome_ledger_incomplete")
    keys = {
        (row["lane"], str(row["opening_id"]), int(row["game"]["challenger_player"]))
        for row in rows
    }
    expected_keys = {
        (lane, str(opening), seat)
        for lane in ("A", "B")
        for opening in range(512)
        for seat in (0, 1)
    }
    if keys != expected_keys:
        raise ValueError("outcome_opening_coverage_incomplete")
    _verify_published_games(root, rows)
    if analysis["samples"] != 20_000 or analysis["seed"] != 429:
        raise ValueError("bootstrap_contract_mismatch")
    reconstructed_analysis = bootstrap_paired(rows, samples=20_000, seed=429)
    for key in (
        "primary_mean_B_minus_A",
        "primary_95_percentile_interval",
        "B_opponent_score_mean",
        "B_opponent_score_95_percentile_interval",
        "per_opening",
    ):
        if analysis[key] != reconstructed_analysis[key]:
            raise ValueError(f"analysis_reconstruction_mismatch:{key}")
    advance = (
        analysis["primary_mean_B_minus_A"] >= 0.03
        and analysis["primary_95_percentile_interval"][0] > 0
        and analysis["B_opponent_score_mean"] >= 0.53
        and analysis["B_opponent_score_95_percentile_interval"][0] > 0.50
    )
    expected_decision = (
        "advance_to_independent_replication" if advance else "stop_normalization_branch"
    )
    if analysis["decision"] != expected_decision:
        raise ValueError("fixed_decision_reconstruction_mismatch")
    matrix_path = data / "per-opening-matrix.json"
    if json.loads(matrix_path.read_text(encoding="utf-8")) != analysis["per_opening"]:
        raise ValueError("published_matrix_mismatch")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    historical_data = root / "docs/data/seed426-canonical-overlap"
    if diagnostics["membership_sha256"] != sha256_file(
        historical_data / "seed427-validation-membership.jsonl.gz"
    ) or diagnostics["prediction_evidence_sha256"] != sha256_file(
        historical_data / "seed427-prediction-evidence.jsonl.gz"
    ):
        raise ValueError("diagnostic_original_membership_or_coefficients_mismatch")
    if (
        sha256_file(diagnostic_evidence_path)
        != diagnostics["diagnostic_evidence_sha256"]
    ):
        raise ValueError("diagnostic_evidence_hash_mismatch")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    diagnostic_checkpoints = {
        "initializer": root / reg["portable_initializer_path"],
        "A_E4": root / training["lanes"]["A"]["checkpoint_dir"] / "E4.npz",
        "B_E4": root / training["lanes"]["B"]["checkpoint_dir"] / "E4.npz",
    }
    for model_name, checkpoint_path in diagnostic_checkpoints.items():
        if sha256_file(checkpoint_path) != diagnostics["checkpoints"][model_name]:
            raise ValueError(f"diagnostic_checkpoint_mismatch:{model_name}")
    with gzip.open(diagnostic_evidence_path, "rt", encoding="utf-8") as stream:
        diagnostic_rows = [json.loads(line) for line in stream if line.strip()]
    diagnostic_models = {"initializer", "A_E4", "B_E4"}
    if {row["model"] for row in diagnostic_rows} != diagnostic_models:
        raise ValueError("diagnostic_model_coverage_invalid")
    for model_name in diagnostic_models:
        rows_for_model = [row for row in diagnostic_rows if row["model"] == model_name]
        if len(rows_for_model) != 14946:
            raise ValueError(f"diagnostic_row_count_invalid:{model_name}")
        reconstructed = aggregate(rows_for_model)
        if reconstructed != diagnostics["aggregates"][model_name]:
            raise ValueError(f"diagnostic_aggregate_mismatch:{model_name}")
    if (
        diagnostics.get("classification")
        != "retrospective diagnostics; not checkpoint selection or arena gate"
    ):
        raise ValueError("diagnostic_classification_invalid")
    return {
        "valid": True,
        "complete": True,
        "decision": analysis["decision"],
        "repository_root": str(root),
    }


def _verify_published_games(root: Path, rows: list[dict[str, Any]]) -> None:
    suite_path = (
        root / "docs/data/seed429-canonical-policy-normalization/openings.jsonl"
    )
    suite = _rows(suite_path)
    for item in rows:
        game_row = item["game"]
        opening_index = int(item["opening_id"])
        if int(game_row["opening_index"]) != opening_index:
            raise ValueError("ledger_game_opening_index_mismatch")
        seat = int(game_row["challenger_player"])
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in suite[opening_index]["prefix_moves"]]
        for relative in prefix:
            if (
                game.over()
                or relative not in game.possible_moves()
                or not game.move(game.pit_index(relative))
            ):
                raise ValueError("ledger_opening_prefix_invalid")
        if suites.canonical_key(game.to_state()) != suites.canonical_key(
            suite[opening_index]["state"]
        ):
            raise ValueError("ledger_opening_identity_invalid")
        trajectory = [
            int(move) for move in str(game_row["trajectory"]).split(",") if move
        ]
        for move in trajectory:
            if (
                game.over()
                or not 0 <= move < 12
                or game.pit_owner(move) != game.current_player
                or not game.move(move)
            ):
                raise ValueError("ledger_trajectory_invalid")
        if not game.over() or len(trajectory) != int(game_row["game_length"]):
            raise ValueError("ledger_trajectory_incomplete")
        margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
        winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
        score = 1.0 if winner == "challenger" else 0.5 if winner == "draw" else 0.0
        if (
            margin != int(game_row["margin"])
            or winner != game_row["winner"]
            or score != float(item["opponent_score"])
        ):
            raise ValueError("ledger_terminal_outcome_mismatch")


def _verify_training(
    root: Path, reg: dict[str, Any], training_path: Path, runtime_path: Path
) -> bool:
    results = json.loads(training_path.read_text(encoding="utf-8"))
    if results.get("registration_sha256") != sha256_file(
        root / "docs/data/seed429-canonical-policy-normalization/registration.json"
    ):
        return False
    for lane in ("A", "B"):
        lane_result = results["lanes"][lane]
        if lane_result["optimizer_updates"] != 1052:
            return False
        for epoch, digest in lane_result["epoch_hashes"].items():
            candidate = root / lane_result["checkpoint_dir"] / f"E{epoch}.npz"
            if sha256_file(candidate) != digest:
                return False
    expected = reg["training"]["expected_A_epoch_hashes"]
    if results["lanes"]["A"]["epoch_hashes"] != expected:
        return False
    if (
        sha256_file(root / results["lanes"]["A"]["checkpoint_dir"] / "E4.npz")
        != "836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd"
    ):
        return False
    binding = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        binding.get("registration_sha256")
        != sha256_file(
            root / "docs/data/seed429-canonical-policy-normalization/registration.json"
        )
        or binding.get("training_results_sha256") != sha256_file(training_path)
        or binding.get("suite_sha256") != sha256_file(root / reg["suite_path"])
    ):
        return False
    opponent = root / binding["opponent"]
    for name, digest in binding["opponent_files"].items():
        if sha256_file(opponent / name) != digest:
            return False
    if binding["opponent_files"] != reg["frozen_runtime"]["opponent_files"]:
        return False
    for lane in ("A", "B"):
        artifact = root / binding["candidates"][lane]["artifact"]
        if (
            binding["candidates"][lane]["checkpoint_sha256"]
            != results["lanes"][lane]["epoch_hashes"]["4"]
        ):
            return False
        for name, digest in binding["candidates"][lane]["artifact_files"].items():
            if sha256_file(artifact / name) != digest:
                return False
        if (
            binding["candidates"][lane]["artifact_files"]["model.npz"]
            != results["lanes"][lane]["epoch_hashes"]["4"]
        ):
            return False
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.root, require_complete=not args.preparation_only),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
