"""Read-only preparation for Seed 599 replay-coverage scaling.

This module authenticates the published #580/#585 evidence before extracting
targets.  It contains no training or arena launch path.
"""

from __future__ import annotations

import hashlib
import json
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import train, varied_opponent_pilot as pilot
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.verify_broader_opening_coverage_results import (
    verify as verify_585,
)
from ml.alphazero_lite.verify_varied_opponent_pilot import verify as verify_580

ROOT = Path(__file__).resolve().parents[2]
SEED = 599
PHASE_COUNTS = {"above_32": 512, "17_to_32": 1024, "at_most_16": 512}
CONTROL_COUNTS = {"above_32": 64, "17_to_32": 128, "at_most_16": 64}
FREEZE_580 = "8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36"
EVIDENCE_580 = "7c8d32ced1fd56ae71f261b9dcbc482b5248a62ce825409513bb9ad78aed2c0c"
EVIDENCE_585 = "e1cf047b1e8587d673ae711c11e29eb9125240391d945a581a27ce47ce37cd75"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def phase(stones: int) -> str:
    if stones > 32:
        return "above_32"
    if stones >= 17:
        return "17_to_32"
    return "at_most_16"


def authenticated_records(
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify both published archives and return all completed retained rows."""
    result_580 = verify_580(root, FREEZE_580, EVIDENCE_580)
    result_585 = verify_585(
        root / "docs/data/broader-opening-coverage-pilot", EVIDENCE_585
    )
    records: list[dict[str, Any]] = []
    for relative, source in (
        ("docs/data/varied-opponent-coverage-pilot/pairs", "#580"),
        ("docs/data/broader-opening-coverage-pilot/pairs", "#585"),
    ):
        for path in sorted((root / relative).glob("*.json")):
            artifact_sha = digest(path)
            payload = json.loads(path.read_bytes())
            for game in payload["result"]["games"]:
                if (
                    game.get("game_status") != "complete"
                    or game.get("game_completed") is not True
                ):
                    raise ValueError("source_game_not_completed")
                for row in game["rows"]:
                    state = row["canonical_state"]
                    legal = KalahGame.from_state(state).possible_moves()
                    target = pilot.reconstruct_policy_target(row, legal_moves=legal)
                    encoded_policy = np.asarray(target, dtype="<f4")
                    records.append(
                        {
                            "source": source,
                            "source_artifact": str(path.relative_to(root)),
                            "source_artifact_sha256": artifact_sha,
                            "game_id": str(row["game_id"]),
                            "opening_id": str(row["opening_id"]),
                            "row_id": str(row["row_id"]),
                            "move_index": int(row["move_index"]),
                            "stone_count": int(row["stone_count"]),
                            "phase": phase(int(row["stone_count"])),
                            "canonical_identity": canonical_identity(state),
                            "float32_identity": production_float32_identity(state),
                            "state": list(row["state"]),
                            "canonical_state": state,
                            "legal_moves": legal,
                            "target": [float(value) for value in target],
                            "target_sha256": hashlib.sha256(
                                encoded_policy.tobytes()
                            ).hexdigest(),
                            "teacher_evidence": {
                                key: row[key]
                                for key in (
                                    "teacher_source",
                                    "target_visits",
                                    "target_temperature",
                                    "exact_selected_action",
                                    "exact_optimal_actions",
                                    "exact_action_margins",
                                    "exact_legal_moves",
                                    "exact_network_priors",
                                    "exact_active_pit_stones",
                                    "exact_root_player",
                                    "exact_solver_calls",
                                )
                                if key in row
                            },
                        }
                    )
    return records, {
        "#580": result_580,
        "#585": result_585,
        "source_artifacts": {
            str(path.relative_to(root)): digest(path)
            for base in (
                "docs/data/varied-opponent-coverage-pilot/pairs",
                "docs/data/broader-opening-coverage-pilot/pairs",
            )
            for path in sorted((root / base).glob("*.json"))
        },
    }


def deduplicate(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    by_identity: dict[str, dict[str, Any]] = {}
    target_sets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        identity = record["canonical_identity"]
        target_sets[identity].add(record["target_sha256"])
        prior = by_identity.get(identity)
        if prior is None or (record["source"], record["game_id"], record["row_id"]) < (
            prior["source"],
            prior["game_id"],
            prior["row_id"],
        ):
            by_identity[identity] = record
    conflicts = sum(len(targets) > 1 for targets in target_sets.values())
    return by_identity, conflicts


def select_membership(
    by_identity: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in by_identity.values():
        buckets[record["phase"]].append(record)
    selected: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    for name, count in PHASE_COUNTS.items():
        ordered = sorted(
            buckets[name],
            key=lambda row: hashlib.sha256(
                f"seed599:{SEED}:{name}:{row['canonical_identity']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < count:
            raise ValueError(
                f"insufficient_phase_positions:{name}:{len(ordered)}:{count}"
            )
        selected.extend(ordered[:count])
        control.extend(ordered[: CONTROL_COUNTS[name]])
    selected.sort(key=lambda row: (row["phase"], row["canonical_identity"]))
    control.sort(key=lambda row: (row["phase"], row["canonical_identity"]))
    validate_membership(selected, control)
    return selected, control


def validate_membership(
    selected: list[dict[str, Any]], control: list[dict[str, Any]]
) -> None:
    if Counter(row["phase"] for row in selected) != Counter(PHASE_COUNTS):
        raise ValueError("treatment_phase_count_mismatch")
    if Counter(row["phase"] for row in control) != Counter(CONTROL_COUNTS):
        raise ValueError("control_phase_count_mismatch")
    for rows in (selected, control):
        if len({row["canonical_identity"] for row in rows}) != len(rows):
            raise ValueError("canonical_identity_duplicate")
        if len({row["float32_identity"] for row in rows}) != len(rows):
            raise ValueError("float32_identity_duplicate")
        for row in rows:
            validate_target_record(row)
    if not {row["canonical_identity"] for row in control} <= {
        row["canonical_identity"] for row in selected
    }:
        raise ValueError("control_not_nested_in_treatment")


def validate_target_record(row: dict[str, Any]) -> None:
    game = KalahGame.from_state(row["canonical_state"])
    legal = game.possible_moves()
    if legal != row["legal_moves"]:
        raise ValueError("legal_action_index_mismatch")
    target = pilot.reconstruct_policy_target(
        {
            **row["teacher_evidence"],
            "stone_count": row["stone_count"],
            "player": row["canonical_state"]["current_player"],
            "move_index": row["move_index"],
        },
        legal_moves=legal,
    )
    if not np.array_equal(
        np.asarray(target, dtype="<f4"), np.asarray(row["target"], dtype="<f4")
    ):
        raise ValueError("target_reconstruction_mismatch")
    if (
        len(target) != 6
        or abs(sum(target) - 1.0) > 1e-6
        or any(target[action] != 0 for action in range(6) if action not in legal)
    ):
        raise ValueError("target_legal_index_validation_failed")


def generate_openings(
    selected: list[dict[str, Any]], *, count: int = 1024
) -> list[dict[str, Any]]:
    """Generate unique legal 12-ply starts, rejecting training-state overlap."""
    training_ids = {row["canonical_identity"] for row in selected}
    rng = random.Random(SEED)
    openings: list[dict[str, Any]] = []
    starts: set[str] = set()
    attempt = 0
    while len(openings) < count:
        attempt += 1
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        moves = []
        local = random.Random(rng.randrange(2**63))
        for _ in range(12):
            legal = game.possible_moves()
            if not legal or game.over():
                break
            move = local.choice(legal)
            moves.append(int(move))
            game.move(game.pit_index(move))
        state = game.to_state()
        canonical = canonical_identity(state)
        f32 = production_float32_identity(state)
        if (
            canonical in training_ids
            or canonical in starts
            or any(row["float32_identity"] == f32 for row in selected)
        ):
            continue
        starts.add(canonical)
        openings.append(
            {
                "opening_id": f"seed599-opening-{len(openings):04d}",
                "prefix_moves": moves,
                "state": state,
                "canonical_identity": canonical,
                "float32_identity": f32,
                "generation_attempt": attempt,
            }
        )
    return openings


def verify_parent_and_parameter_groups(root: Path) -> dict[str, Any]:
    parent = root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz"
    if (
        digest(parent)
        != "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
    ):
        raise ValueError("parent_checkpoint_hash_mismatch")
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, parent):
        raise ValueError("parent_checkpoint_tensor_mismatch")
    serving_files = {
        "metadata": (
            root
            / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/metadata.json",
            "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
        ),
        "model": (
            parent,
            "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
        ),
        "weights": (
            root
            / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/weights.json",
            "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
        ),
        "search_policy": (
            root
            / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/search_policy.json",
            "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
        ),
        "native_probe": (
            root / "model-artifact/runtime/kalah_v1_tablebase",
            "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        ),
        "tablebase": (
            root / "model-artifact/runtime/kalah_v1_21.kvtb",
            "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
        ),
    }
    for name, (path, expected) in serving_files.items():
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"serving_evaluation_asset_hash_mismatch:{name}")
    trainable = (
        "input_layer.",
        "residual_layers.",
        "policy_hidden_layer.",
        "policy_head.",
    )
    frozen = ("value_hidden_layer.", "value_head.")
    groups = {"trainable": [], "frozen": []}
    snapshots = {}
    for name, parameter in model.named_parameters():
        should_train = name.startswith(trainable)
        if not should_train and not name.startswith(frozen):
            raise ValueError(f"unregistered_parameter_group:{name}")
        parameter.requires_grad_(should_train)
        groups["trainable" if should_train else "frozen"].append(name)
        if not should_train:
            snapshots[name] = parameter.detach().clone()
    for name, before in snapshots.items():
        if not train.torch.equal(dict(model.named_parameters())[name].detach(), before):
            raise ValueError("frozen_parameter_mutated_during_preflight")
    return {
        "parent_checkpoint_sha256": digest(parent),
        "serving_evaluation_asset_sha256": {
            name: value for name, (_path, value) in serving_files.items()
        },
        "parameter_groups": groups,
        "frozen_parameters_unchanged": True,
        "optimizer_schedule": "Adam fixed lr 1e-5; no scheduler; 512 updates",
    }


def batch_schedule(size: int) -> list[list[int]]:
    """Freeze all 512 zero-based minibatches for an arm."""
    rng = random.Random(SEED)
    schedule: list[list[int]] = []
    while len(schedule) < 512:
        indexes = list(range(size))
        rng.shuffle(indexes)
        schedule.extend(indexes[start : start + 32] for start in range(0, size, 32))
    return schedule[:512]


def build(root: Path = ROOT) -> dict[str, Any]:
    records, verification = authenticated_records(root)
    verification["training_contract"] = verify_parent_and_parameter_groups(root)
    identities, conflicts = deduplicate(records)
    if conflicts:
        raise ValueError(f"conflicting_duplicate_targets:{conflicts}")
    provenance_groups = sorted({(row["source"], row["opening_id"]) for row in records})
    heldout_groups = set(
        sorted(
            provenance_groups,
            key=lambda group: hashlib.sha256(
                f"seed599-heldout:{SEED}:{group[0]}:{group[1]}".encode()
            ).hexdigest(),
        )[: (len(provenance_groups) + 4) // 5]
    )
    heldout_ids = {
        row["canonical_identity"]
        for row in records
        if (row["source"], row["opening_id"]) in heldout_groups
    }
    training_candidates = {
        identity: row
        for identity, row in identities.items()
        if identity not in heldout_ids
        and (row["source"], row["opening_id"]) not in heldout_groups
    }
    selected, control = select_membership(training_candidates)
    heldout = sorted(
        (row for identity, row in identities.items() if identity in heldout_ids),
        key=lambda row: (row["phase"], row["canonical_identity"]),
    )
    if {row["canonical_identity"] for row in selected} & {
        row["canonical_identity"] for row in heldout
    }:
        raise ValueError("training_holdout_state_overlap")
    openings = generate_openings(selected)
    phase_rows = Counter(record["phase"] for record in records)
    distinct_games = {record["game_id"] for record in records}
    distinct_openings = {record["opening_id"] for record in records}
    return {
        "schema": "seed599_replay_coverage_preparation_v1",
        "status": "prepared-training-and-evaluation-not-authorized",
        "seed": SEED,
        "inventory": {
            "eligible_rows": len(records),
            "unique_canonical_states": len(identities),
            "unique_float32_states": len(
                {row["float32_identity"] for row in identities.values()}
            ),
            "phase_bin_row_counts": dict(phase_rows),
            "phase_bin_unique_counts": dict(
                Counter(row["phase"] for row in identities.values())
            ),
            "source_games": len(distinct_games),
            "opening_groups": len(distinct_openings),
            "conflicting_targets": conflicts,
        },
        "source_verification": verification,
        "training": {
            "treatment": selected,
            "control": control,
            "treatment_counts": dict(Counter(row["phase"] for row in selected)),
            "control_counts": dict(Counter(row["phase"] for row in control)),
            "common": {
                "parent_checkpoint_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
                "architecture": {
                    "model_type": "residual_v3",
                    "hidden_sizes": [96, 3],
                    "input_size": 27,
                },
                "trainable_prefixes": [
                    "input_layer.",
                    "residual_layers.",
                    "policy_hidden_layer.",
                    "policy_head.",
                ],
                "frozen_parameter_groups": ["value_hidden_layer.", "value_head."],
                "optimizer": {
                    "name": "Adam",
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "scheduler": "none",
                },
                "updates": 512,
                "batch_size": 32,
                "gradient_clip_norm": 1.0,
                "rng_seed": SEED,
                "batch_order": "Python random.Random(599) shuffle of local membership indexes per pass; consume batches of 32 and cycle passes until exactly 512 updates; see frozen batch-schedule files",
                "runtime": {
                    "device": "cpu",
                    "torch_threads": 1,
                    "deterministic_algorithms": True,
                },
                "loss": "masked policy cross-entropy only; no value loss",
            },
        },
        "heldout": {
            "source_opening_groups": [list(group) for group in sorted(heldout_groups)],
            "states": heldout,
            "states_by_phase": dict(Counter(row["phase"] for row in heldout)),
            "provenance_disjoint_from_training": True,
        },
        "evaluation": {
            "opening_seed": SEED,
            "opening_count": len(openings),
            "opening_generation": "independent legal uniform-random 12-ply prefixes; unique canonical starts; reject starts matching selected training states in canonical or production float32 identity",
            "opening_manifest": openings,
            "comparisons": ["A versus serving P-E4", "B versus serving P-E4"],
            "value_evaluator": "serving P-E4, unchanged and common to all policies",
            "search": "serving PUCT configuration including native exact-root behavior",
            "match_settings": {
                "simulations_per_side": 128,
                "c_puct": 1.25,
                "max_plies": 200,
                "workers": 1,
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
                "noise": False,
                "exact_root_enabled": True,
                "exact_root_threshold": 16,
                "exact_root_objective": "final_score_margin",
                "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
                "exact_leaf_solving": False,
            },
            "serving_p_e4": {
                "metadata_sha256": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
                "model_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
                "weights_sha256": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
                "search_policy_sha256": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
                "native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
                "tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
            },
            "seating": "paired seat-swapped games for every opening and comparison",
            "bootstrap": {
                "method": "paired opening-cluster bootstrap",
                "resamples": 20000,
                "seed": SEED,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
            },
            "thresholds": {
                "strength": "lower 95% paired bootstrap bound for B minus A seat-averaged score > -0.02; report both arms versus P-E4",
                "generalization": "B held-out masked policy cross-entropy on provenance-held-out source opening groups must be lower than A; report paired opening-cluster 95% interval",
            },
            "trajectory_overlap": "report exact canonical and float32 overlap counts against both training arms; not yet observed",
        },
    }


def write_preparation(root: Path = ROOT) -> dict[str, Any]:
    output = build(root)
    destination = root / "docs/data/seed599-replay-coverage"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("preparation.json", output),
        ("treatment-membership.json", output["training"]["treatment"]),
        ("control-membership.json", output["training"]["control"]),
        ("evaluation-openings.json", output["evaluation"]["opening_manifest"]),
        ("heldout-membership.json", output["heldout"]["states"]),
    ):
        (destination / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    common = output["training"]["common"]
    for arm, rows in (
        ("A", output["training"]["control"]),
        ("B", output["training"]["treatment"]),
    ):
        schedule = batch_schedule(len(rows))
        configuration = {
            "schema": "seed599_training_configuration_v1",
            "status": "frozen-pretraining-not-authorized",
            "arm": arm,
            "membership_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
            "batch_schedule_sha256": hashlib.sha256(
                canonical_bytes(schedule)
            ).hexdigest(),
            "states": len(rows),
            "phase_counts": dict(Counter(row["phase"] for row in rows)),
            "target_construction": "reconstruct_policy_target from authenticated #580/#585 teacher evidence",
            **common,
        }
        (destination / f"training-{arm}.json").write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n"
        )
        (destination / f"batch-schedule-{arm}.json").write_text(
            json.dumps(schedule, separators=(",", ":")) + "\n"
        )
    contract = {
        key: value
        for key, value in output["evaluation"].items()
        if key != "opening_manifest"
    }
    contract["opening_manifest_sha256"] = hashlib.sha256(
        canonical_bytes(output["evaluation"]["opening_manifest"])
    ).hexdigest()
    contract["status"] = "frozen-evaluation-not-authorized"
    (destination / "evaluation-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return output


def read_only_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Rebuild all registrations in memory and compare without writing artifacts."""
    directory = root / "docs/data/seed599-replay-coverage"
    expected = build(root)
    persisted = json.loads((directory / "preparation.json").read_bytes())
    if canonical_bytes(persisted) != canonical_bytes(expected):
        raise ValueError("preparation_manifest_mismatch")
    memberships = {
        "treatment-membership.json": expected["training"]["treatment"],
        "control-membership.json": expected["training"]["control"],
        "evaluation-openings.json": expected["evaluation"]["opening_manifest"],
        "heldout-membership.json": expected["heldout"]["states"],
    }
    for name, value in memberships.items():
        if canonical_bytes(
            json.loads((directory / name).read_bytes())
        ) != canonical_bytes(value):
            raise ValueError(f"preparation_artifact_mismatch:{name}")
    common = expected["training"]["common"]
    for arm, rows in (
        ("A", expected["training"]["control"]),
        ("B", expected["training"]["treatment"]),
    ):
        schedule = batch_schedule(len(rows))
        configuration = json.loads((directory / f"training-{arm}.json").read_bytes())
        expected_configuration = {
            "schema": "seed599_training_configuration_v1",
            "status": "frozen-pretraining-not-authorized",
            "arm": arm,
            "membership_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
            "batch_schedule_sha256": hashlib.sha256(
                canonical_bytes(schedule)
            ).hexdigest(),
            "states": len(rows),
            "phase_counts": dict(Counter(row["phase"] for row in rows)),
            "target_construction": "reconstruct_policy_target from authenticated #580/#585 teacher evidence",
            **common,
        }
        if canonical_bytes(configuration) != canonical_bytes(expected_configuration):
            raise ValueError(f"training_configuration_mismatch:{arm}")
        if canonical_bytes(
            json.loads((directory / f"batch-schedule-{arm}.json").read_bytes())
        ) != canonical_bytes(schedule):
            raise ValueError(f"training_batch_schedule_mismatch:{arm}")
    expected_contract = {
        key: value
        for key, value in expected["evaluation"].items()
        if key != "opening_manifest"
    }
    expected_contract["opening_manifest_sha256"] = hashlib.sha256(
        canonical_bytes(expected["evaluation"]["opening_manifest"])
    ).hexdigest()
    expected_contract["status"] = "frozen-evaluation-not-authorized"
    if canonical_bytes(
        json.loads((directory / "evaluation-contract.json").read_bytes())
    ) != canonical_bytes(expected_contract):
        raise ValueError("evaluation_contract_mismatch")
    return {
        "passed": True,
        "training_or_evaluation_started": False,
        "eligible_rows": expected["inventory"]["eligible_rows"],
        "distinct_states": expected["inventory"]["unique_canonical_states"],
        "treatment_states": len(expected["training"]["treatment"]),
        "control_states": len(expected["training"]["control"]),
        "opening_groups": expected["evaluation"]["opening_count"],
        "heldout_states": len(expected["heldout"]["states"]),
        "frozen_parameter_groups_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight"))
    args = parser.parse_args()
    result = write_preparation() if args.command == "prepare" else read_only_preflight()
    print(json.dumps(result, indent=2, sort_keys=True))
