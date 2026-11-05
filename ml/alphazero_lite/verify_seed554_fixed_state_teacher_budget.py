"""Independent, read-only verification of seed554 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed554_fixed_state_teacher_budget as protocol
from ml.alphazero_lite import seed553_fixed_state_budget as teacher
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed554-fixed-state-teacher-budget"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_unique_occurrences(rows: list[dict[str, Any]], expected: int) -> None:
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise ValueError("verifier_occurrence_multiplicity_mismatch")


def require_nonpolicy_source_row(
    source: dict[str, Any], derivative: dict[str, Any], arm: str, index: int
) -> None:
    if set(derivative) != set(source):
        raise ValueError(f"verifier_nonpolicy_field_set_mismatch:{arm}:{index}")
    if any(
        canon_value(derivative[key]) != canon_value(source[key])
        for key in source
        if key != "policy"
    ):
        raise ValueError(f"verifier_nonpolicy_source_row_mutation:{arm}:{index}")


def resolve_search_state(
    row: dict[str, Any], state_map: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    """Bind archived index and seed to the reconstructed canonical ordering."""
    canonical_id = row.get("canonical_state_sha256")
    if not isinstance(canonical_id, str):
        raise ValueError("verifier_search_state_identity_missing")
    item = state_map.get(canonical_id)
    if item is None:
        raise ValueError("verifier_search_state_not_in_population")
    if row.get("state_index") != item["state_index"]:
        raise ValueError("verifier_search_state_index_mismatch")
    if row.get("production_float32_sha256") != item["float32"]:
        raise ValueError("verifier_search_float32_identity_mismatch")
    return item, 554_000 + item["state_index"]


def reconstruct_source_population() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Rebuild occurrence identities from raw trajectories without publisher helpers."""
    corpus = ROOT / "docs/data/seed549-matched-selfplay"
    manifest = json.loads((corpus / "control-source-manifest.json").read_text())
    shard_by_id = {int(item["shard"]): item for item in manifest["shards"]}
    identities = [
        json.loads(line)
        for line in (corpus / "control-training-row-identities.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    occurrence_by_key = {}
    for identity in identities:
        source = identity["source"]
        key = (
            int(source["shard"]),
            int(source["game_index"]),
            int(source["move_index"]),
        )
        if key in occurrence_by_key:
            raise ValueError("verifier_duplicate_source_occurrence")
        occurrence_by_key[key] = identity

    def canonical_identity(state: dict[str, Any]) -> str:
        values = (
            state["player_pits"]
            + state["opponent_pits"]
            + [state["player_store"], state["opponent_store"], state["current_player"]]
        )
        return digest(json.dumps(values, separators=(",", ":")).encode())

    records = []
    for shard in range(protocol.SHARDS):
        manifest_row = shard_by_id[shard]
        path = ROOT / manifest_row["path"]
        if digest(path.read_bytes()) != manifest_row["sha256"]:
            raise ValueError(f"verifier_control_shard_hash_mismatch:{shard}")
        raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        row_numbers = {id(row): index for index, row in enumerate(raw_rows)}
        games: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            games[int(row["game_index"])].append(row)
        for game_id, game_rows in games.items():
            ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
            game = KalahGame.from_state(ordered[0]["game_start_state"])
            moves = ordered[0]["game_moves"]
            for row, move in zip(ordered, moves, strict=True):
                state = game.to_state()
                key = (shard, game_id, int(row["move_index"]))
                identity = occurrence_by_key.get(key)
                if identity is not None:
                    encoded = np.asarray(
                        encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
                    )
                    canonical_id = canonical_identity(state)
                    float32_id = digest(encoded.tobytes())
                    source = identity["source"]
                    if (
                        source.get("path") != path.relative_to(ROOT).as_posix()
                        or int(source.get("row_number", -1)) != row_numbers[id(row)]
                        or identity.get("canonical_state_sha256") != canonical_id
                        or identity.get("production_float32_sha256") != float32_id
                        or int(identity.get("active_stones", -1))
                        != sum(state["player_pits"] + state["opponent_pits"])
                    ):
                        raise ValueError("verifier_source_occurrence_identity_mismatch")
                    if (
                        encoded.tobytes()
                        != np.asarray(row["state"], dtype=np.float32).tobytes()
                    ):
                        raise ValueError(
                            "verifier_source_input_reconstruction_mismatch"
                        )
                    records.append(
                        {
                            "occurrence": identity,
                            "row": row,
                            "state": state,
                            "canonical": canonical_id,
                            "float32": float32_id,
                            "encoded": encoded,
                        }
                    )
                if not game.move(game.pit_index(int(move))):
                    raise ValueError("verifier_source_trajectory_replay_failed")
            if not game.over():
                raise ValueError("verifier_source_game_not_terminal")
    if len(records) != protocol.OCCURRENCES:
        raise ValueError("verifier_source_occurrence_total_mismatch")
    records.sort(
        key=lambda item: (
            int(item["occurrence"]["source"]["shard"]),
            int(item["occurrence"]["source"]["row_number"]),
        )
    )
    if [item["occurrence"]["row_id"] for item in records] != [
        item["row_id"] for item in identities
    ]:
        raise ValueError("verifier_source_occurrence_order_mismatch")
    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        prior = unique.get(item["canonical"])
        if prior is not None and json.dumps(
            prior["state"], sort_keys=True
        ) != json.dumps(item["state"], sort_keys=True):
            raise ValueError("verifier_canonical_state_collision")
        unique.setdefault(item["canonical"], item)
    states = sorted(
        unique.values(), key=lambda item: (item["canonical"], item["float32"])
    )
    for index, item in enumerate(states):
        item["state_index"] = index
        item["seed"] = 554_000 + index
    return records, states, identities


def validate_search_evidence(
    record: dict[str, Any], state: dict[str, Any], budget: int, seed: int
) -> None:
    if record.get("simulations") != budget or record.get("seed") != seed:
        raise ValueError("verifier_search_protocol_mismatch")
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    visits = record.get("visits")
    if (
        not isinstance(visits, list)
        or len(visits) != 6
        or any(type(value) is not int or value < 0 for value in visits)
    ):
        raise ValueError("verifier_search_visit_shape_invalid")
    if sum(visits) != budget or any(
        visits[action] != 0 for action in set(range(6)) - set(legal)
    ):
        raise ValueError("verifier_search_visit_accounting_invalid")
    expected = teacher.self_play.build_policy_target(
        np.asarray(visits, dtype=np.int64), legal_moves=legal, temperature=1.0
    )
    if (
        np.asarray(expected, dtype=np.float32).tobytes()
        != np.asarray(record.get("target"), dtype=np.float32).tobytes()
    ):
        raise ValueError("verifier_search_target_reconstruction_mismatch")
    if record.get("legal_moves") != legal or record.get("root_visit_count") != budget:
        raise ValueError("verifier_search_root_accounting_mismatch")
    selected = min(
        action for action in legal if visits[action] == max(visits[a] for a in legal)
    )
    if record.get("selected_move") != selected:
        raise ValueError("verifier_search_tie_rule_mismatch")
    children = record.get("root_children")
    if not isinstance(children, dict) or set(children) != {
        str(action) for action in legal
    }:
        raise ValueError("verifier_search_root_children_mismatch")
    for action in legal:
        child = children[str(action)]
        if not isinstance(child, dict) or child.get("visits") != visits[action]:
            raise ValueError("verifier_search_child_visit_mismatch")
        if any(
            not isinstance(child.get(field), (int, float))
            or not np.isfinite(child[field])
            for field in ("prior", "q_value")
        ):
            raise ValueError("verifier_search_child_numeric_mismatch")


def verify(*, rerun_searches: bool = False) -> dict[str, Any]:
    if rerun_searches:
        protocol.live_generation_preflight()
    protocol.preflight()
    environment_binding = json.loads(
        (OUT / "execution-environment-binding.json").read_text()
    )
    seed553_registration_path = (
        ROOT / "docs/data/seed553-fixed-state-budget/registration.json"
    )
    if protocol.file_sha(seed553_registration_path) != environment_binding.get(
        "recorded_numerical_environment_source_sha256"
    ):
        raise ValueError("verifier_recorded_environment_source_hash_mismatch")
    seed553_registration = json.loads(seed553_registration_path.read_text())
    recorded_runtime = seed553_registration["runtime"]
    actual_runtime = environment_binding["runtime"]
    for field, recorded_field in (
        ("python", "python"),
        ("platform", "platform"),
        ("numpy", "numpy"),
        ("torch", "torch"),
    ):
        if actual_runtime.get(field) != recorded_runtime.get(recorded_field):
            raise ValueError(f"verifier_recorded_environment_mismatch:{field}")
    if environment_binding.get("resource_limits") != seed553_registration.get(
        "resource_limits"
    ):
        raise ValueError("verifier_recorded_resource_limits_mismatch")
    records, states, identities = reconstruct_source_population()
    state_map = {item["canonical"]: item for item in states}
    evidence_path = OUT / "searches.jsonl"
    searches: dict[tuple[str, int], dict[str, Any]] = {}
    evaluator = None
    if rerun_searches:
        evaluator = teacher.self_play.CheckpointEvaluator(
            protocol.INITIALIZER, input_encoding="kalah_v3", cache_size=0
        )
    for line in evidence_path.read_text().splitlines():
        row = json.loads(line)
        canonical_id = row["canonical_state_sha256"]
        item, seed = resolve_search_state(row, state_map)
        state = item["state"]
        budget = int(row["simulations"])
        validate_search_evidence(row["search"], state, budget, seed)
        key = canonical_id, budget
        if key in searches:
            raise ValueError("verifier_duplicate_search")
        if rerun_searches:
            fresh = teacher.run_search(
                state, simulations=budget, seed=seed, evaluator=evaluator
            )
            for field in (
                "visits",
                "target",
                "selected_move",
                "legal_moves",
                "root_visit_count",
                "root_q_value",
                "root_evaluation_value",
                "root_children",
                "evaluator_calls",
            ):
                left, right = row["search"].get(field), fresh.get(field)
                if field in {"target", "root_q_value", "root_evaluation_value"}:
                    if (
                        np.asarray(left, dtype=np.float64).tobytes()
                        != np.asarray(right, dtype=np.float64).tobytes()
                    ):
                        raise ValueError(
                            f"verifier_search_replay_mismatch:{field}:{canonical_id}:{budget}"
                        )
                elif left != right:
                    raise ValueError(
                        f"verifier_search_replay_mismatch:{field}:{canonical_id}:{budget}"
                    )
        searches[key] = row
    searched = [
        item
        for item in states
        if sum(item["state"]["player_pits"] + item["state"]["opponent_pits"]) > 16
    ]
    expected_order = [
        (item["canonical"], budget)
        for item in searched
        for budget in ([128, 512] if item["state_index"] % 2 == 0 else [512, 128])
    ]
    observed_order = [
        (
            json.loads(line)["canonical_state_sha256"],
            int(json.loads(line)["simulations"]),
        )
        for line in evidence_path.read_text().splitlines()
    ]
    if observed_order != expected_order:
        raise ValueError("verifier_alternating_search_order_mismatch")
    if len(searches) != len(searched) * 2:
        raise ValueError("verifier_search_coverage_mismatch")
    mapping = [
        json.loads(line)
        for line in (OUT / "occurrence-to-search.jsonl").read_text().splitlines()
    ]
    require_unique_occurrences(mapping, protocol.OCCURRENCES)
    if len(mapping) != len(records):
        raise ValueError("verifier_occurrence_mapping_length_mismatch")
    for index, (mapped, expected) in enumerate(zip(mapping, records, strict=True)):
        source = expected["occurrence"]["source"]
        if (
            mapped.get("occurrence_index") != index
            or mapped.get("row_id") != expected["occurrence"]["row_id"]
        ):
            raise ValueError("verifier_occurrence_order_mismatch")
        if (
            mapped.get("canonical_state_sha256") != expected["canonical"]
            or mapped.get("production_float32_sha256") != expected["float32"]
        ):
            raise ValueError("verifier_occurrence_state_identity_mismatch")
        if mapped.get("source") != source:
            raise ValueError("verifier_occurrence_source_binding_mismatch")
    derivatives = {}
    for arm, budget in (("control", 128), ("treatment", 512)):
        schedule_path = OUT / f"{arm}-training-schedule.json"
        if (
            schedule_path.read_bytes()
            != (protocol.CORPUS / "control-training-schedule.json").read_bytes()
        ):
            raise ValueError(f"verifier_schedule_not_original:{arm}")
        rows = []
        for shard in range(protocol.SHARDS):
            path = OUT / arm / f"shard-{shard:03d}.jsonl"
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        if len(rows) != protocol.OCCURRENCES:
            raise ValueError(f"verifier_derivative_occurrence_count:{arm}")
        derivatives[arm] = rows
        for index, source in enumerate(records):
            derivative_row = rows[index]
            source_row = source["row"]
            require_nonpolicy_source_row(source_row, derivative_row, arm, index)
            target = np.asarray(rows[index]["policy"], dtype=np.float32)
            stone_count = mapping[index]["active_stones"]
            if stone_count <= 16:
                if (
                    target.tobytes()
                    != np.asarray(source["row"]["policy"], dtype=np.float32).tobytes()
                ):
                    raise ValueError(f"verifier_low_stone_target_changed:{arm}:{index}")
            else:
                key = source["canonical"], budget
                search = searches.get(key)
                if search is None:
                    raise ValueError(
                        f"verifier_changed_target_without_search:{arm}:{index}"
                    )
                expected_target = np.asarray(
                    search["search"]["target"], dtype=np.float32
                )
                if target.tobytes() != expected_target.tobytes():
                    raise ValueError(
                        f"verifier_search_target_binding_mismatch:{arm}:{index}"
                    )
                if mapping[index].get(f"{arm}_search_sha256") != digest(
                    json.dumps(search, sort_keys=True, separators=(",", ":")).encode()
                ):
                    raise ValueError(
                        f"verifier_occurrence_search_receipt_mismatch:{arm}:{index}"
                    )
    for index in range(protocol.OCCURRENCES):
        for key in ("state", "value", "legal_moves"):
            if canon_value(derivatives["control"][index].get(key)) != canon_value(
                derivatives["treatment"][index].get(key)
            ):
                raise ValueError(f"verifier_nonpolicy_row_mismatch:{key}:{index}")
    from ml.alphazero_lite import train

    loaded = {}
    for arm in ("control", "treatment"):
        loaded[arm] = train.load_jsonl_replay(
            [
                OUT / arm / f"shard-{shard:03d}.jsonl"
                for shard in range(protocol.SHARDS)
            ],
            weights=[1] * protocol.SHARDS,
            policy_target_mode="default",
            value_target_mode="default",
            include_policy_loss_weights=True,
        )
    for name, tensor_index in (("input", 0), ("value", 2), ("policy_weight", 4)):
        if (
            np.asarray(loaded["control"][tensor_index]).tobytes()
            != np.asarray(loaded["treatment"][tensor_index]).tobytes()
        ):
            raise ValueError(f"verifier_production_tensor_mismatch:{name}")
    masks = [
        train.legal_mask_matrix_for_encoded_states(loaded[arm][0])
        for arm in ("control", "treatment")
    ]
    if np.asarray(masks[0]).tobytes() != np.asarray(masks[1]).tobytes():
        raise ValueError("verifier_production_tensor_mismatch:legal_mask")
    expected_tensor_hashes = {
        arm: {
            "input": digest(np.ascontiguousarray(loaded[arm][0]).tobytes()),
            "policy": digest(np.ascontiguousarray(loaded[arm][1]).tobytes()),
            "value": digest(np.ascontiguousarray(loaded[arm][2]).tobytes()),
            "weight": digest(np.ascontiguousarray(loaded[arm][4]).tobytes()),
            "legal_mask": digest(np.ascontiguousarray(masks[index]).tobytes()),
        }
        for index, arm in enumerate(("control", "treatment"))
    }
    tensor_summary = json.loads((OUT / "derivative-tensor-summary.json").read_text())
    if tensor_summary.get("tensors") != expected_tensor_hashes:
        raise ValueError("verifier_derivative_tensor_hash_mismatch")
    policy_a = np.asarray(loaded["control"][1], dtype=np.float32)
    policy_b = np.asarray(loaded["treatment"][1], dtype=np.float32)
    stones = np.asarray([row["active_stones"] for row in mapping])
    total_variation = 0.5 * np.abs(
        policy_a.astype(np.float64) - policy_b.astype(np.float64)
    ).sum(axis=1)
    top_a = np.argmax(
        np.where(np.asarray(masks[0], dtype=bool), policy_a, -1.0), axis=1
    )
    top_b = np.argmax(
        np.where(np.asarray(masks[0], dtype=bool), policy_b, -1.0), axis=1
    )
    differences = {}
    for bucket, selected in (
        ("pooled", np.ones(protocol.OCCURRENCES, dtype=bool)),
        (">32", stones > 32),
        ("17-32", (stones >= 17) & (stones <= 32)),
        ("<=16", stones <= 16),
    ):
        indexes = np.flatnonzero(selected)
        differences[bucket] = {
            "occurrences": len(indexes),
            "policy_targets_bitwise_different": int(
                sum(policy_a[i].tobytes() != policy_b[i].tobytes() for i in indexes)
            ),
            "highest_target_action_differences": int(
                np.count_nonzero(top_a[indexes] != top_b[indexes])
            ),
            "total_variation": {
                "mean": float(np.mean(total_variation[indexes]))
                if len(indexes)
                else None,
                "median": float(np.median(total_variation[indexes]))
                if len(indexes)
                else None,
                "maximum": float(np.max(total_variation[indexes]))
                if len(indexes)
                else None,
            },
        }
    if differences != json.loads((OUT / "target-differences.json").read_text()):
        raise ValueError("verifier_target_difference_summary_mismatch")
    binding_path = OUT / "training-input-binding.json"
    if binding_path.exists():
        training_binding = json.loads(binding_path.read_text())
        if training_binding.get("label_generation_freeze_sha256") != protocol.file_sha(
            OUT / "label-generation-freeze.json"
        ):
            raise ValueError("verifier_training_binding_freeze_mismatch")
        if training_binding.get(
            "execution_environment_binding_sha256"
        ) != protocol.file_sha(OUT / "execution-environment-binding.json"):
            raise ValueError("verifier_training_environment_binding_mismatch")
        if training_binding.get(
            "seed553_recorded_environment_sha256"
        ) != protocol.file_sha(
            ROOT / "docs/data/seed553-fixed-state-budget/registration.json"
        ):
            raise ValueError("verifier_training_source_environment_binding_mismatch")
        corrections_sha = protocol.file_sha(OUT / "source-corrections.jsonl")
        correction_parent = None
        corrected_source_hash = training_binding.get("source_corrections_sha256")
        for number in range(1, 10):
            correction_path = (
                OUT / f"training-input-binding-correction-{number:02d}.json"
            )
            if not correction_path.is_file():
                break
            correction = json.loads(correction_path.read_text())
            if (
                correction.get("training_input_binding_sha256")
                != protocol.file_sha(binding_path)
                or correction.get("previous_source_corrections_sha256")
                != corrected_source_hash
                or (
                    correction_parent is not None
                    and correction.get("parent_correction_sha256") != correction_parent
                )
            ):
                raise ValueError("verifier_training_input_correction_chain_mismatch")
            corrected_source_hash = correction.get("source_corrections_sha256")
            correction_parent = protocol.file_sha(correction_path)
        if corrected_source_hash != corrections_sha:
            raise ValueError("verifier_execution_correction_binding_mismatch")
        if training_binding.get("attempt_ledger_sha256") != protocol.file_sha(
            OUT / "attempts.jsonl"
        ):
            raise ValueError("verifier_execution_correction_binding_mismatch")
        if training_binding.get("derivative_verification_sha256") != protocol.file_sha(
            OUT / "derivative-verification.json"
        ):
            raise ValueError("verifier_derivative_verification_binding_mismatch")
        expected_derivatives = {
            f"{arm}/{shard:03d}": protocol.file_sha(
                OUT / arm / f"shard-{shard:03d}.jsonl"
            )
            for arm in ("control", "treatment")
            for shard in range(protocol.SHARDS)
        }
        expected_schedules = {
            arm: protocol.file_sha(OUT / f"{arm}-training-schedule.json")
            for arm in ("control", "treatment")
        }
        if (
            training_binding.get("derivative_sha256") != expected_derivatives
            or training_binding.get("schedule_sha256") != expected_schedules
            or training_binding.get("occurrence_mapping_sha256")
            != protocol.file_sha(OUT / "occurrence-to-search.jsonl")
            or training_binding.get("search_evidence_sha256")
            != protocol.file_sha(OUT / "searches.jsonl")
            or training_binding.get("rows") != protocol.OCCURRENCES
        ):
            raise ValueError("verifier_training_input_binding_content_mismatch")
    training_summary = None
    if (OUT / "control-updates.jsonl").exists() or (
        OUT / "treatment-updates.jsonl"
    ).exists():
        if (
            not (OUT / "control-updates.jsonl").is_file()
            or not (OUT / "treatment-updates.jsonl").is_file()
        ):
            raise ValueError("verifier_partial_training_trajectory")
        from ml.alphazero_lite import verify_seed549_training as trajectory_verifier

        # Reuse the independent #550 mathematical replay, but point it at the
        # paired derivatives and the seed554-produced traces/checkpoints.
        trajectory_verifier.ROOT = ROOT
        trajectory_verifier.OUT = OUT
        trajectory_verifier.CORPUS = OUT
        trajectory_verifier.configure_torch_threads()
        for arm in ("control", "treatment"):
            trajectory_verifier.replay(arm)
        control_updates = [
            json.loads(line)
            for line in (OUT / "control-updates.jsonl").read_text().splitlines()
        ]
        for number, record in enumerate(control_updates, start=1):
            if (
                record.get("production_equivalent") is not True
                or record.get("production_parameter_state_before_sha256")
                != record.get("parameter_state_before_sha256")
                or record.get("production_parameter_state_after_sha256")
                != record.get("parameter_state_after_sha256")
                or record.get("production_optimizer_state_before_sha256")
                != record.get("optimizer_state_before_sha256")
                or record.get("production_optimizer_state_after_sha256")
                != record.get("optimizer_state_after_sha256")
            ):
                raise ValueError(
                    f"verifier_control_production_equivalence_mismatch:{number}"
                )
        gate_path = OUT / "control-production-equivalence.jsonl"
        if gate_path.exists():
            gate_rows = [
                json.loads(line) for line in gate_path.read_text().splitlines()
            ]
            if len(gate_rows) != 108:
                raise ValueError("verifier_control_production_gate_row_count_mismatch")
            for number, (gate, record) in enumerate(
                zip(gate_rows, control_updates, strict=True), start=1
            ):
                expected_comparisons = {
                    "parameter_before_matches": gate.get(
                        "production_parameter_state_before_sha256"
                    )
                    == record.get("parameter_state_before_sha256"),
                    "parameter_after_matches": gate.get(
                        "production_parameter_state_after_sha256"
                    )
                    == record.get("parameter_state_after_sha256"),
                    "optimizer_before_matches": gate.get(
                        "production_optimizer_state_before_sha256"
                    )
                    == record.get("optimizer_state_before_sha256"),
                    "optimizer_after_matches": gate.get(
                        "production_optimizer_state_after_sha256"
                    )
                    == record.get("optimizer_state_after_sha256"),
                }
                if (
                    gate.get("update") != number
                    or gate.get("comparisons") != expected_comparisons
                    or gate.get("production_equivalent") is not True
                    or not all(expected_comparisons.values())
                    or gate.get("direct_parameter_state_before_sha256")
                    != record.get("parameter_state_before_sha256")
                    or gate.get("direct_parameter_state_after_sha256")
                    != record.get("parameter_state_after_sha256")
                    or gate.get("direct_optimizer_state_before_sha256")
                    != record.get("optimizer_state_before_sha256")
                    or gate.get("direct_optimizer_state_after_sha256")
                    != record.get("optimizer_state_after_sha256")
                ):
                    raise ValueError(
                        f"verifier_control_production_gate_mismatch:{number}"
                    )
            derivation_path = OUT / "control-production-gate-derivation-01.json"
            if derivation_path.exists():
                derivation = json.loads(derivation_path.read_text())
                if (
                    derivation.get("production_gate_sha256")
                    != protocol.file_sha(gate_path)
                    or derivation.get("source_control_trace_sha256")
                    != protocol.file_sha(OUT / "control-updates.jsonl")
                    or derivation.get("production_gate_rows") != len(gate_rows)
                    or derivation.get("status") != "all_108_updates_passed"
                    or protocol.file_sha(
                        ROOT / derivation["seed549_training_executed_source_snapshot"]
                    )
                    != derivation.get("seed549_training_executed_source_sha256")
                    or protocol.file_sha(ROOT / "ml/alphazero_lite/seed549_training.py")
                    != derivation.get("seed549_training_restored_source_sha256")
                ):
                    raise ValueError(
                        "verifier_control_production_gate_derivation_mismatch"
                    )
        training_summary = {
            "control_updates": 108,
            "treatment_updates": 108,
            "production_gate": "all C128 parameter and optimizer states replayed against train_one_epoch",
        }
    development_summary = (
        _verify_development() if (OUT / "development-results.json").exists() else None
    )
    return {
        "occurrences": len(identities),
        "unique_canonical_states": len(states),
        "paired_searches": len(searches),
        "derivative_rows_each": protocol.OCCURRENCES,
        "rerun_searches": rerun_searches,
        "training_input_binding_sha256": protocol.file_sha(binding_path)
        if binding_path.exists()
        else None,
        "training": training_summary,
        "development": development_summary,
    }


def _verify_development() -> dict[str, Any]:
    from ml.alphazero_lite import train

    receipt = json.loads((protocol.CORPUS / "post-generation-receipt.json").read_text())
    products = receipt.get("products", {})
    for arm in ("control", "treatment"):
        manifest_path = protocol.CORPUS / f"{arm}-source-manifest.json"
        if products.get(f"{arm}-source-manifest.json") != protocol.file_sha(
            manifest_path
        ):
            raise ValueError(f"verifier_development_manifest_receipt_mismatch:{arm}")
        manifest = json.loads(manifest_path.read_text())
        for shard in manifest["shards"]:
            if shard.get("partition") != "development":
                continue
            if protocol.file_sha(ROOT / shard["path"]) != shard["sha256"]:
                raise ValueError(
                    f"verifier_development_shard_hash_mismatch:{shard['path']}"
                )
    result = json.loads((OUT / "development-results.json").read_text())
    evidence = [
        json.loads(line)
        for line in (OUT / "development-predictions.jsonl").read_text().splitlines()
    ]
    membership = [
        json.loads(line)
        for line in (protocol.CORPUS / "shared-development-membership.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line).get("included")
    ]
    if (
        len(membership) != protocol.DEVELOPMENT
        or len(evidence) != 3 * protocol.DEVELOPMENT
    ):
        raise ValueError("verifier_development_occurrence_count_mismatch")
    names = {
        "initializer": protocol.INITIALIZER,
        "C-final": OUT / "control-update-108.npz",
        "T-final": OUT / "treatment-update-108.npz",
    }
    development_shards: dict[str, list[dict[str, Any]]] = {}
    for endpoint, checkpoint in names.items():
        rows = [row for row in evidence if row.get("endpoint") == endpoint]
        if len(rows) != protocol.DEVELOPMENT:
            raise ValueError(f"verifier_development_endpoint_count:{endpoint}")
        network = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(network, checkpoint)
        network.eval()
        encoded_rows = []
        for index, (row, member) in enumerate(zip(rows, membership, strict=True)):
            if (
                row.get("occurrence") != index
                or row.get("source_evaluator") != member["source_evaluator"]
                or row.get("cohort") != member["stone_bucket"]
            ):
                raise ValueError(
                    f"verifier_development_membership_mismatch:{endpoint}:{index}"
                )
            source = member["source"]
            path = source["path"]
            if path not in development_shards:
                development_shards[path] = [
                    json.loads(line) for line in (ROOT / path).read_text().splitlines()
                ]
            raw = development_shards[path]
            training_row = raw[int(source["row_number"])]
            encoded = np.asarray(training_row["state"], dtype=np.float32)
            if digest(encoded.tobytes()) != row.get("float32_input_sha256"):
                raise ValueError(
                    f"verifier_development_input_identity_mismatch:{endpoint}:{index}"
                )
            if np.asarray(
                training_row["policy"], dtype=np.float32
            ).tobytes() != np.asarray(
                row["target_policy"], dtype=np.float32
            ).tobytes() or float(training_row["value"]) != float(
                row["target_actual_value"]
            ):
                raise ValueError(
                    f"verifier_development_target_mismatch:{endpoint}:{index}"
                )
            encoded_rows.append(encoded)
        development_inputs = np.asarray(encoded_rows, dtype=np.float32)
        with torch.no_grad():
            logits_batch, value_batch = network(torch.from_numpy(development_inputs))
        legal_batch = train.legal_mask_matrix_for_encoded_states(
            development_inputs
        ).astype(bool)
        for index, row in enumerate(rows):
            legal = legal_batch[index]
            if not np.array_equal(legal, np.asarray(row["legal_mask"], dtype=bool)):
                raise ValueError(
                    f"verifier_development_legal_mask_mismatch:{endpoint}:{index}"
                )
            logits_np = logits_batch.numpy()[index].astype(np.float64)
            logits_np[~legal] = -1e9
            logp = logits_np - np.logaddexp.reduce(logits_np)
            probs = np.exp(logp)
            if (
                not np.array_equal(
                    logits_np,
                    np.asarray(row["policy_logits_legal_masked"], dtype=np.float64),
                )
                or not np.allclose(
                    probs, row["policy_probabilities_legal_masked"], rtol=0, atol=1e-7
                )
                or not np.isclose(
                    float(value_batch[index].item()),
                    float(row["value_prediction"]),
                    rtol=0,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    f"verifier_development_prediction_mismatch:{endpoint}:{index}"
                )
    reconstructed: dict[str, Any] = {"metrics": {}, "effects": {}}
    sources = ("control", "treatment", "combined")
    cohorts = ("pooled", ">32", "17-32", "<=16")
    for endpoint in names:
        rows = [row for row in evidence if row["endpoint"] == endpoint]
        ce, mse = [], []
        for row in rows:
            logits = np.asarray(row["policy_logits_legal_masked"], dtype=np.float64)
            logp = logits - np.logaddexp.reduce(logits)
            target = np.asarray(row["target_policy"], dtype=np.float64)
            ce.append(float(-(target * logp).sum()))
            mse.append(
                (float(row["value_prediction"]) - float(row["target_actual_value"]))
                ** 2
            )
        endpoint_metrics: dict[str, Any] = {}
        for source in sources:
            endpoint_metrics[source] = {}
            for cohort in cohorts:
                selected = [
                    i
                    for i, row in enumerate(rows)
                    if (source == "combined" or row["source_evaluator"] == source)
                    and (cohort == "pooled" or row["cohort"] == cohort)
                ]
                groups: dict[str, list[int]] = {}
                for i in selected:
                    groups.setdefault(rows[i]["float32_input_sha256"], []).append(i)
                if not selected:
                    endpoint_metrics[source][cohort] = None
                    continue
                endpoint_metrics[source][cohort] = {
                    "occurrences": len(selected),
                    "distinct_float32_inputs": len(groups),
                    "exposure": {
                        "policy_ce": float(np.mean([ce[i] for i in selected])),
                        "actual_value_mse": float(np.mean([mse[i] for i in selected])),
                    },
                    "equal_input": {
                        "policy_ce": float(
                            np.mean(
                                [
                                    np.mean([ce[i] for i in indexes])
                                    for indexes in groups.values()
                                ]
                            )
                        ),
                        "actual_value_mse": float(
                            np.mean(
                                [
                                    np.mean([mse[i] for i in indexes])
                                    for indexes in groups.values()
                                ]
                            )
                        ),
                    },
                }
        reconstructed["metrics"][endpoint] = endpoint_metrics
    for endpoint, baseline in (
        ("T-final", "C-final"),
        ("C-final", "initializer"),
        ("T-final", "initializer"),
    ):
        key = f"{endpoint}_minus_{baseline}"
        reconstructed["effects"][key] = {}
        for source in sources:
            reconstructed["effects"][key][source] = {}
            for cohort in cohorts:
                reconstructed["effects"][key][source][cohort] = {}
                left = reconstructed["metrics"][endpoint][source][cohort]
                right = reconstructed["metrics"][baseline][source][cohort]
                for weighting in ("exposure", "equal_input"):
                    if left is None or right is None:
                        expected_effect = None
                    else:
                        expected_effect = {
                            metric: left[weighting][metric] - right[weighting][metric]
                            for metric in ("policy_ce", "actual_value_mse")
                        }
                    reconstructed["effects"][key][source][cohort][weighting] = (
                        expected_effect
                    )
    published_metrics = result.get("metrics", {})
    published_effects = result.get("effects", {})
    for endpoint in names:
        for source in sources:
            for cohort in cohorts:
                expected = reconstructed["metrics"][endpoint][source][cohort]
                actual = published_metrics[endpoint]["cohorts"][source][cohort]
                if expected is None or actual is None:
                    if expected is not actual:
                        raise ValueError("verifier_development_metric_null_mismatch")
                else:
                    for weighting in ("exposure", "equal_input"):
                        for metric in ("policy_ce", "actual_value_mse"):
                            if not np.isclose(
                                expected[weighting][metric],
                                actual[weighting][metric],
                                rtol=0,
                                atol=1e-12,
                            ):
                                raise ValueError(
                                    f"verifier_development_metric_mismatch:{endpoint}:{source}:{cohort}:{weighting}:{metric}"
                                )
    for key in reconstructed["effects"]:
        for source in sources:
            for cohort in cohorts:
                for weighting in ("exposure", "equal_input"):
                    expected = reconstructed["effects"][key][source][cohort][weighting]
                    actual = published_effects[key][source][cohort][weighting]
                    if expected is None or actual is None:
                        if expected is not actual:
                            raise ValueError(
                                "verifier_development_effect_null_mismatch"
                            )
                    elif any(
                        not np.isclose(expected[m], actual[m], rtol=0, atol=1e-12)
                        for m in ("policy_ce", "actual_value_mse")
                    ):
                        raise ValueError("verifier_development_effect_mismatch")
    return {
        "occurrences": len(membership),
        "endpoint_predictions": len(evidence),
        "checkpoints_replayed": len(names),
        "results_sha256": protocol.file_sha(OUT / "development-results.json"),
        "independent_metrics_and_effects": "verified",
    }


def canon_value(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-searches", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(verify(rerun_searches=args.rerun_searches), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
