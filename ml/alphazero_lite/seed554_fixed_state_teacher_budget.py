"""Paired fixed-state teacher-budget training ablation for seed554.

This module authenticates #549 by its bounded source and occurrence bindings,
creates temperature-one policy derivatives, and drives the unchanged #549
training schedule. It deliberately does not invoke the historical tablebase
runtime: only roots above 16 stones are searched and PUCT exact-leaf solving is
disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed553_fixed_state_budget as teacher
from ml.alphazero_lite import train
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed549-matched-selfplay"
OUT = ROOT / "docs/data/seed554-fixed-state-teacher-budget"
INITIALIZER = ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
SHARDS = 6
OCCURRENCES = 13_818
DEVELOPMENT = 4_738
SCHEDULE_SHA256 = ""
ARMS = ("control", "treatment")  # C128 and T512 in the training primitive.


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def append_durable(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def authenticate_inputs() -> dict[str, Any]:
    """Validate source manifests, six shard hashes, occurrences, and schedule."""
    manifest_path = CORPUS / "control-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training_shards = [
        item for item in manifest["shards"] if item["partition"] == "training"
    ]
    if len(training_shards) != SHARDS:
        raise ValueError("control_training_shard_count_mismatch")
    if sum(int(item["rows"]) for item in training_shards) != OCCURRENCES:
        raise ValueError("control_training_manifest_occurrence_total_mismatch")
    for item in training_shards:
        if file_sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"control_shard_hash_mismatch:{item['path']}")
    identity_path = CORPUS / "control-training-row-identities.jsonl"
    identities = [
        json.loads(line) for line in identity_path.read_text().splitlines() if line
    ]
    if (
        len(identities) != OCCURRENCES
        or len({row["row_id"] for row in identities}) != OCCURRENCES
    ):
        raise ValueError("control_occurrence_population_mismatch")
    receipt = json.loads((CORPUS / "post-generation-receipt.json").read_text())
    receipt_products = receipt.get("products", {})
    for product in (
        "control-source-manifest.json",
        "control-training-row-identities.jsonl",
        "control-training-schedule.json",
        "shared-development-membership.jsonl",
    ):
        if receipt_products.get(product) != file_sha(CORPUS / product):
            raise ValueError(f"published_receipt_product_binding_mismatch:{product}")
    schedule_path = CORPUS / "control-training-schedule.json"
    schedule = json.loads(schedule_path.read_text())
    if schedule.get("updates") != 108 or schedule.get("exposures") != 54_896:
        raise ValueError("control_schedule_totals_mismatch")
    if len(schedule.get("batches", [])) != 108:
        raise ValueError("control_schedule_update_count_mismatch")
    cursor = 0
    batch_sizes = ([512] * 26 + [412]) * 4
    for update, (batch, size) in enumerate(
        zip(schedule["batches"], batch_sizes, strict=True), start=1
    ):
        indices = batch.get("row_occurrence_indices")
        if (
            batch.get("update") != update
            or not isinstance(indices, list)
            or len(indices) != size
            or batch.get("start_exposure") != cursor
            or batch.get("end_exposure_exclusive") != cursor + size
            or any(
                type(index) is not int or not 0 <= index < OCCURRENCES
                for index in indices
            )
        ):
            raise ValueError(f"control_schedule_batch_mapping_mismatch:{update}")
        cursor += size
    if cursor != 54_896:
        raise ValueError("control_schedule_exposure_cursor_mismatch")
    if file_sha(INITIALIZER) != INITIALIZER_SHA256:
        raise ValueError("initializer_sha256_mismatch")
    development = [
        json.loads(line)
        for line in (CORPUS / "shared-development-membership.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    if sum(bool(row.get("included")) for row in development) != DEVELOPMENT:
        raise ValueError("shared_development_membership_mismatch")
    return {
        "control_source_manifest_sha256": file_sha(manifest_path),
        "control_training_shards": [
            {"path": item["path"], "sha256": item["sha256"], "rows": item["rows"]}
            for item in training_shards
        ],
        "occurrence_identity_sha256": file_sha(identity_path),
        "occurrences": len(identities),
        "published_receipt_sha256": file_sha(CORPUS / "post-generation-receipt.json"),
        "receipt": receipt,
        "schedule_sha256": file_sha(schedule_path),
        "updates": 108,
        "exposures": 54_896,
        "development_membership_sha256": file_sha(
            CORPUS / "shared-development-membership.jsonl"
        ),
        "development_occurrences": DEVELOPMENT,
        "initializer_sha256": file_sha(INITIALIZER),
        "historical_verifier_scope": "bounded source, occurrence, receipt, schedule, and initializer authentication; full historical runtime verifier not claimed",
    }


def _states_by_occurrence() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    identities = [
        json.loads(line)
        for line in (CORPUS / "control-training-row-identities.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    occurrence_by_key = {}
    for item in identities:
        s = item["source"]
        key = (int(s["shard"]), int(s["game_index"]), int(s["move_index"]))
        if key in occurrence_by_key:
            raise ValueError("duplicate_authenticated_source_occurrence")
        occurrence_by_key[key] = item
    records: list[dict[str, Any]] = []
    for shard in range(SHARDS):
        path = CORPUS / "control" / f"shard-{shard:03d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        row_numbers = {id(row): index for index, row in enumerate(rows)}
        games: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            games[int(row["game_index"])].append(row)
        for game_id, game_rows in games.items():
            game_rows.sort(key=lambda row: int(row["move_index"]))
            game = KalahGame.from_state(game_rows[0]["game_start_state"])
            moves = game_rows[0]["game_moves"]
            for row, move in zip(game_rows, moves, strict=True):
                state = game.to_state()
                key = (shard, game_id, int(row["move_index"]))
                occurrence = occurrence_by_key.get(key)
                if occurrence is not None:
                    encoded = np.asarray(
                        encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
                    )
                    source = occurrence["source"]
                    canonical_id = teacher.canonical_state_id(state)
                    float32_id = sha(encoded.tobytes())
                    if (
                        source.get("path") != path.relative_to(ROOT).as_posix()
                        or int(source.get("row_number", -1)) != row_numbers[id(row)]
                        or occurrence.get("canonical_state_sha256") != canonical_id
                        or occurrence.get("production_float32_sha256") != float32_id
                        or int(occurrence.get("active_stones", -1))
                        != sum(state["player_pits"] + state["opponent_pits"])
                    ):
                        raise ValueError("authenticated_occurrence_semantics_mismatch")
                    if (
                        encoded.tobytes()
                        != np.asarray(row["state"], dtype=np.float32).tobytes()
                    ):
                        raise ValueError("production_input_reconstruction_mismatch")
                    records.append(
                        {
                            "occurrence": occurrence,
                            "row": row,
                            "state": state,
                            "canonical": canonical_id,
                            "float32": float32_id,
                            "encoded": encoded,
                        }
                    )
                if not game.move(game.pit_index(int(move))):
                    raise ValueError("control_source_replay_failed")
            if not game.over():
                raise ValueError("control_source_game_not_terminal")
    if len(records) != OCCURRENCES:
        raise ValueError(f"reconstructed_occurrence_count_mismatch:{len(records)}")
    records.sort(
        key=lambda item: (
            int(item["occurrence"]["source"]["shard"]),
            int(item["occurrence"]["source"]["row_number"]),
        )
    )
    if [item["occurrence"]["row_id"] for item in records] != [
        item["row_id"] for item in identities
    ]:
        raise ValueError("source_occurrence_order_differs_from_frozen_schedule")
    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        # A canonical collision is explicitly checked against exact state bytes;
        # production float32 equality alone never merges canonical states.
        prior = unique.get(item["canonical"])
        if prior is not None and canon(prior["state"]) != canon(item["state"]):
            raise ValueError("canonical_identity_collision")
        unique.setdefault(item["canonical"], item)
    states = sorted(
        unique.values(), key=lambda item: (item["canonical"], item["float32"])
    )
    for index, item in enumerate(states):
        item["state_index"] = index
        item["seed"] = 554_000 + index
    return records, states, identities


def freeze() -> None:
    if OUT.exists() and any(path.name != "README.md" for path in OUT.iterdir()):
        raise FileExistsError("seed554_output_directory_not_empty")
    OUT.mkdir(parents=True, exist_ok=True)
    auth = authenticate_inputs()
    environment_binding = build_execution_environment_binding()
    write_json(OUT / "input-authentication.json", auth)
    write_json(OUT / "execution-environment-binding.json", environment_binding)
    environment_binding_sha256 = file_sha(OUT / "execution-environment-binding.json")
    source_files = [
        "ml/alphazero_lite/seed554_fixed_state_teacher_budget.py",
        "ml/alphazero_lite/verify_seed554_fixed_state_teacher_budget.py",
        "ml/alphazero_lite/seed553_fixed_state_budget.py",
        "ml/alphazero_lite/seed549_training.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/test_seed554_fixed_state_teacher_budget.py",
        "docs/data/seed554-fixed-state-teacher-budget/README.md",
    ]
    freeze = {
        "schema": "seed554-label-generation-freeze-v1",
        "sources": {name: file_sha(ROOT / name) for name in source_files},
        "inputs": auth,
        "execution_environment_binding_sha256": environment_binding_sha256,
        "label_protocol": {
            "roots": "all unique canonical training states with >16 active stones",
            "budgets": [128, 512],
            "seed": "554000 + sorted canonical-state index; paired",
            "order": "alternate budget-first by state index",
            "fresh_tree_rng": True,
            "policy_temperature": 1.0,
            "search": teacher.search_config(128),
            "exact_leaf_solve": "disabled",
            "exact_root_dispatch": "not used",
            "low_stone_policy": "copy source target byte-for-byte",
        },
        "training": {
            "schedule": "unchanged control-training-schedule.json; 108 updates / 54896 exposures",
            "architecture": "residual_v3 hidden (96,3), kalah_v3 27 inputs",
            "objective": "legal weighted policy CE + 0.5 * Huber(value, delta=1)",
            "optimizer": "fresh Adam per arm, lr=1e-4, betas=(0.9,0.999), eps=1e-8, no decay/scheduler, clip=1.0",
        },
        "resource_limits": teacher.RESOURCE_LIMITS,
        "bounded_authentication_note": auth["historical_verifier_scope"],
    }
    write_json(OUT / "label-generation-freeze.json", freeze)
    write_json(
        OUT / "registration.json",
        {
            "label_generation_freeze_sha256": file_sha(
                OUT / "label-generation-freeze.json"
            ),
            "input_authentication_sha256": file_sha(OUT / "input-authentication.json"),
            "execution_environment_binding_sha256": environment_binding_sha256,
        },
    )


def build_execution_environment_binding() -> dict[str, Any]:
    """Build the prospective environment contract from the supported #553 runtime."""
    registration_path = ROOT / "docs/data/seed553-fixed-state-budget/registration.json"
    registration_sha256 = file_sha(registration_path)
    recorded = json.loads(registration_path.read_text())
    runtime = recorded["runtime"]
    return {
        "schema": "seed554-execution-environment-binding-v2",
        "recorded_numerical_environment_source": "docs/data/seed553-fixed-state-budget/registration.json",
        "recorded_numerical_environment_source_sha256": registration_sha256,
        "runtime": {
            "python": runtime["python"],
            "platform": runtime["platform"],
            "numpy": runtime["numpy"],
            "torch": runtime["torch"],
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": 1,
            "torch_interop_threads": 1,
        },
        "resource_limits": teacher.RESOURCE_LIMITS,
        "label_generation_launcher": teacher.RESOURCE_LAUNCHER,
        "training_launcher_requirement": "same systemd resource bounds and affinity/nice; OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=BLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1",
        "scope": "source authentication and CPU PUCT only; no exact solver/native queries",
    }


def verify_published_environment_binding() -> str:
    binding_path = OUT / "execution-environment-binding.json"
    if not binding_path.is_file():
        raise ValueError("execution_environment_binding_missing")
    observed_sha = file_sha(binding_path)
    published_hashes = []
    for path, field in (
        (OUT / "label-generation-freeze.json", "execution_environment_binding_sha256"),
        (OUT / "registration.json", "execution_environment_binding_sha256"),
        (OUT / "training-input-binding.json", "execution_environment_binding_sha256"),
    ):
        if path.is_file():
            digest = json.loads(path.read_text()).get(field)
            if digest is not None:
                published_hashes.append(digest)
    if not published_hashes or any(
        digest != observed_sha for digest in published_hashes
    ):
        raise ValueError("execution_environment_binding_hash_mismatch")
    return observed_sha


def preflight() -> dict[str, Any]:
    reg_path = OUT / "registration.json"
    reg = json.loads(reg_path.read_text())
    freeze = json.loads((OUT / "label-generation-freeze.json").read_text())
    if reg["label_generation_freeze_sha256"] != file_sha(
        OUT / "label-generation-freeze.json"
    ):
        raise ValueError("label_generation_freeze_binding_mismatch")
    if reg["input_authentication_sha256"] != file_sha(
        OUT / "input-authentication.json"
    ):
        raise ValueError("input_authentication_binding_mismatch")
    verify_published_environment_binding()
    current = authenticate_inputs()
    if current != freeze["inputs"]:
        raise ValueError("frozen_input_authentication_changed")
    corrections_path = OUT / "source-corrections.jsonl"
    corrections: dict[tuple[str, str], str] = {}
    previous = "0" * 64
    if corrections_path.exists():
        for sequence, line in enumerate(corrections_path.read_text().splitlines(), 1):
            correction = json.loads(line)
            if (
                correction.get("sequence") != sequence
                or correction.get("previous_sha256") != previous
                or correction.get("registration_sha256") != file_sha(reg_path)
            ):
                raise ValueError("source_correction_chain_invalid")
            previous = sha(line.encode() + b"\n")
            corrections[(correction["source_path"], correction["before_sha256"])] = (
                correction["after_sha256"]
            )
    for name, expected_digest in freeze["sources"].items():
        actual_digest = file_sha(ROOT / name)
        candidate = expected_digest
        while candidate != actual_digest and (name, candidate) in corrections:
            candidate = corrections[(name, candidate)]
        if candidate != actual_digest:
            raise ValueError(f"frozen_source_changed:{name}")
    return freeze


def _live_cgroup_limits() -> dict[str, str | None]:
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(
            line.split("::", 1)[1] for line in memberships if line.startswith("0::")
        )
        directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    except (OSError, StopIteration):
        raise ValueError("generation_cgroup_scope_unavailable") from None
    values = {}
    for filename in ("cpu.max", "cpu.weight", "memory.max"):
        try:
            values[filename] = (directory / filename).read_text().strip()
        except OSError as error:
            raise ValueError(
                f"generation_cgroup_limit_unavailable:{filename}"
            ) from error
    return values


def validate_live_generation_contract(
    binding: dict[str, Any],
    registration: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    expected_runtime = binding["runtime"]
    expected_packages = registration["runtime"]["pip_freeze"]
    if any(
        observed.get(key) != expected_runtime.get(key)
        for key in ("python", "platform", "numpy", "torch")
    ):
        raise ValueError("generation_numerical_runtime_mismatch")
    if observed.get("packages") != expected_packages:
        raise ValueError("generation_dependency_set_mismatch")
    if (
        observed.get("torch_threads") != expected_runtime["torch_threads"]
        or observed.get("torch_interop_threads")
        != expected_runtime["torch_interop_threads"]
    ):
        raise ValueError("generation_torch_thread_mismatch")
    limits = binding["resource_limits"]
    if (
        observed.get("cpu_affinity") != [int(limits["cpu_affinity"])]
        or observed.get("nice") != limits["nice"]
    ):
        raise ValueError("generation_affinity_or_nice_mismatch")
    cgroup = observed.get("cgroup_limits", {})
    if (
        cgroup.get("cpu.max") != "25000 100000"
        or cgroup.get("cpu.weight") != str(limits["cpu_weight"])
        or cgroup.get("memory.max") != str(limits["memory_max_bytes"])
    ):
        raise ValueError("generation_cgroup_resource_contract_mismatch")


def live_generation_preflight() -> None:
    """Fail before state loading, evaluator construction, writes, or search."""
    verify_published_environment_binding()
    binding_path = OUT / "execution-environment-binding.json"
    binding = json.loads(binding_path.read_text())
    registration_path = ROOT / binding["recorded_numerical_environment_source"]
    if (
        file_sha(registration_path)
        != binding["recorded_numerical_environment_source_sha256"]
    ):
        raise ValueError("generation_recorded_environment_source_hash_mismatch")
    registration = json.loads(registration_path.read_text())
    runtime = binding["runtime"]
    if (sys.version, platform.platform(), np.__version__, torch.__version__) != (
        runtime["python"],
        runtime["platform"],
        runtime["numpy"],
        runtime["torch"],
    ):
        raise ValueError("generation_numerical_runtime_mismatch")
    packages = sorted(
        (
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=str.casefold,
    )
    observed = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "packages": packages,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.nice(0),
        "cgroup_limits": _live_cgroup_limits(),
    }
    validate_live_generation_contract(binding, registration, observed)


def _validate_search(
    record: dict[str, Any], state: dict[str, Any], budget: int, seed: int
) -> None:
    if record.get("simulations") != budget or record.get("seed") != seed:
        raise ValueError("saved_search_protocol_mismatch")
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    visits = record.get("visits")
    if (
        not isinstance(visits, list)
        or len(visits) != 6
        or any(type(x) is not int or x < 0 for x in visits)
    ):
        raise ValueError("saved_search_visit_shape_invalid")
    if sum(visits) != budget:
        raise ValueError("saved_search_visit_total_mismatch")
    if any(visits[action] != 0 for action in set(range(6)) - set(legal)):
        raise ValueError("saved_search_illegal_action_visit")
    expected = teacher.self_play.build_policy_target(
        np.asarray(visits, dtype=np.int64), legal_moves=legal, temperature=1.0
    )
    if (
        np.asarray(expected, dtype=np.float32).tobytes()
        != np.asarray(record.get("target"), dtype=np.float32).tobytes()
    ):
        raise ValueError("saved_search_target_reconstruction_mismatch")
    if record.get("legal_moves") != legal or record.get("root_visit_count") != budget:
        raise ValueError("saved_search_root_evidence_mismatch")
    selected = min(
        action for action in legal if visits[action] == max(visits[a] for a in legal)
    )
    if record.get("selected_move") != selected:
        raise ValueError("saved_search_selected_move_mismatch")
    children = record.get("root_children")
    if not isinstance(children, dict) or set(children) != {
        str(action) for action in legal
    }:
        raise ValueError("saved_search_root_children_mismatch")
    for action in legal:
        child = children[str(action)]
        if not isinstance(child, dict) or child.get("visits") != visits[action]:
            raise ValueError("saved_search_child_visit_mismatch")
        for field in ("prior", "q_value"):
            value = child.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("saved_search_child_numeric_invalid")
    if (
        not isinstance(record.get("evaluator_calls"), int)
        or record["evaluator_calls"] <= 0
    ):
        raise ValueError("saved_search_evaluator_call_count_invalid")


def train_policy_target(visits: list[int], legal: list[int]) -> list[float]:
    state = np.zeros(6, dtype=np.float64)
    for action in legal:
        state[action] = visits[action]
    total = state.sum()
    if total <= 0:
        raise ValueError("empty_teacher_policy_target")
    return (state / total).tolist()


def generate() -> None:
    live_generation_preflight()
    freeze = preflight()
    records, states, _ = _states_by_occurrence()
    search_path = OUT / "searches.jsonl"
    completed = _load_search_ledger(states, search_path)
    evaluator = teacher.self_play.CheckpointEvaluator(
        INITIALIZER, input_encoding="kalah_v3", cache_size=0
    )
    for item in states:
        if sum(item["state"]["player_pits"] + item["state"]["opponent_pits"]) <= 16:
            continue
        for budget in [128, 512] if item["state_index"] % 2 == 0 else [512, 128]:
            key = (item["canonical"], budget)
            if key in completed:
                continue
            search = teacher.run_search(
                item["state"],
                simulations=budget,
                seed=item["seed"],
                evaluator=evaluator,
            )
            row = {
                "state_index": item["state_index"],
                "canonical_state_sha256": item["canonical"],
                "production_float32_sha256": item["float32"],
                "simulations": budget,
                "search": search,
            }
            _validate_search(search, item["state"], budget, item["seed"])
            append_durable(search_path, row)
            completed[key] = row
    _build_derivatives(records, states, completed, freeze)


def _load_search_ledger(
    states: list[dict[str, Any]], search_path: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not search_path.exists():
        return completed
    state_map = {item["canonical"]: item for item in states}
    for line in search_path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        sid, budget = row["canonical_state_sha256"], int(row["simulations"])
        item = state_map.get(sid)
        if item is None or item["state_index"] != row.get("state_index"):
            raise ValueError("saved_search_unknown_or_misindexed_state")
        if row.get("production_float32_sha256") != item["float32"]:
            raise ValueError("saved_search_production_input_binding_mismatch")
        _validate_search(row["search"], item["state"], budget, item["seed"])
        key = (sid, budget)
        if key in completed:
            raise ValueError("saved_search_duplicate")
        completed[key] = row
    return completed


def _build_derivatives(
    records: list[dict[str, Any]],
    states: list[dict[str, Any]],
    searches: dict[tuple[str, int], dict[str, Any]],
    freeze: dict[str, Any],
) -> None:
    map_rows = []
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    state_indices = {item["canonical"]: item["state_index"] for item in states}
    for ordinal, item in enumerate(records):
        source = item["occurrence"]["source"]
        shard = int(source["shard"])
        row = dict(item["row"])
        stones = sum(item["state"]["player_pits"] + item["state"]["opponent_pits"])
        binding = {
            "occurrence_index": ordinal,
            "row_id": item["occurrence"]["row_id"],
            "source": source,
            "canonical_state_sha256": item["canonical"],
            "production_float32_sha256": item["float32"],
            "state_index": state_indices[item["canonical"]],
            "active_stones": stones,
        }
        by_shard[shard].append({"row": row, "binding": binding, "item": item})
        map_rows.append(binding)
    for arm, budget in (("control", 128), ("treatment", 512)):
        for shard in range(SHARDS):
            rows_out = []
            for entry in by_shard[shard]:
                row, item = entry["row"], entry["item"]
                stones = entry["binding"]["active_stones"]
                if stones > 16:
                    evidence = searches[(item["canonical"], budget)]
                    target = np.asarray(evidence["search"]["target"], dtype=np.float32)
                    row["policy"] = target.tolist()
                    entry["binding"][f"{arm}_search_sha256"] = sha(canon(evidence))
                rows_out.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
            path = OUT / arm / f"shard-{shard:03d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(rows_out) + "\n")
        # The source schedule is copied unchanged; only the population labels differ.
        (OUT / f"{arm}-training-schedule.json").write_bytes(
            (CORPUS / "control-training-schedule.json").read_bytes()
        )
        with (OUT / f"{arm}-training-row-identities.jsonl").open("w") as stream:
            for entry in map_rows:
                stream.write(
                    json.dumps(
                        {
                            "row_id": entry["row_id"],
                            "source": entry["source"],
                            "canonical_state_sha256": entry["canonical_state_sha256"],
                            "production_float32_sha256": entry[
                                "production_float32_sha256"
                            ],
                            "active_stones": entry["active_stones"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    with (OUT / "occurrence-to-search.jsonl").open("w") as stream:
        for row in map_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    # Production loader tensor-level identity check, including weights and masks.
    loaded = {}
    for arm in ARMS:
        paths = [OUT / arm / f"shard-{i:03d}.jsonl" for i in range(SHARDS)]
        loaded[arm] = train.load_jsonl_replay(
            paths,
            weights=[1] * SHARDS,
            policy_target_mode="default",
            value_target_mode="default",
            include_policy_loss_weights=True,
        )
    for name, index in (("input", 0), ("value", 2), ("weight", 4)):
        left, right = loaded["control"][index], loaded["treatment"][index]
        if (
            np.asarray(left).shape != np.asarray(right).shape
            or np.asarray(left).tobytes() != np.asarray(right).tobytes()
        ):
            raise ValueError(f"paired_training_tensor_mismatch:{name}")
    mask_a = train.legal_mask_matrix_for_encoded_states(loaded["control"][0])
    mask_b = train.legal_mask_matrix_for_encoded_states(loaded["treatment"][0])
    if np.asarray(mask_a).tobytes() != np.asarray(mask_b).tobytes():
        raise ValueError("paired_training_tensor_mismatch:legal_mask")
    policy_differences = sum(
        np.asarray(loaded["control"][1][index], dtype=np.float32).tobytes()
        != np.asarray(loaded["treatment"][1][index], dtype=np.float32).tobytes()
        for index in range(OCCURRENCES)
    )
    high_policy = np.asarray(loaded["control"][1], dtype=np.float32)
    treatment_policy = np.asarray(loaded["treatment"][1], dtype=np.float32)
    legal_masks = np.asarray(mask_a, dtype=bool)
    stone_counts = np.asarray([row["active_stones"] for row in map_rows])
    tv = 0.5 * np.abs(
        high_policy.astype(np.float64) - treatment_policy.astype(np.float64)
    ).sum(axis=1)
    top_c = np.argmax(np.where(legal_masks, high_policy, -1.0), axis=1)
    top_t = np.argmax(np.where(legal_masks, treatment_policy, -1.0), axis=1)
    target_difference_summary = {}
    for bucket, selection in (
        ("pooled", np.ones(OCCURRENCES, dtype=bool)),
        (">32", stone_counts > 32),
        ("17-32", (stone_counts >= 17) & (stone_counts <= 32)),
        ("<=16", stone_counts <= 16),
    ):
        indices = np.flatnonzero(selection)
        target_difference_summary[bucket] = {
            "occurrences": int(len(indices)),
            "policy_targets_bitwise_different": int(
                sum(
                    high_policy[i].tobytes() != treatment_policy[i].tobytes()
                    for i in indices
                )
            ),
            "highest_target_action_differences": int(
                np.count_nonzero(top_c[indices] != top_t[indices])
            ),
            "total_variation": {
                "mean": float(np.mean(tv[indices])) if len(indices) else None,
                "median": float(np.median(tv[indices])) if len(indices) else None,
                "maximum": float(np.max(tv[indices])) if len(indices) else None,
            },
        }
    write_json(OUT / "target-differences.json", target_difference_summary)
    for index, item in enumerate(records):
        row = map_rows[index]
        if row["active_stones"] <= 16:
            source = np.asarray(item["row"]["policy"], dtype=np.float32)
            # Check the production-loader tensors, not only serialized rows.
            for arm in ARMS:
                if (
                    np.asarray(loaded[arm][1][index], dtype=np.float32).tobytes()
                    != source.tobytes()
                ):
                    raise ValueError("low_stone_target_copy_mismatch")

    def array_hash(value: Any) -> str:
        return sha(np.ascontiguousarray(np.asarray(value)).tobytes())

    write_json(
        OUT / "derivative-tensor-summary.json",
        {
            "ordered_occurrences": OCCURRENCES,
            "policy_target_occurrences_differing": int(policy_differences),
            "tensors": {
                arm: {
                    "input": array_hash(loaded[arm][0]),
                    "policy": array_hash(loaded[arm][1]),
                    "value": array_hash(loaded[arm][2]),
                    "weight": array_hash(loaded[arm][4]),
                    "legal_mask": array_hash(
                        train.legal_mask_matrix_for_encoded_states(loaded[arm][0])
                    ),
                }
                for arm in ARMS
            },
        },
    )
    if policy_differences == 0:
        write_json(
            OUT / "no-op-result.json",
            {
                "status": "no_scheduled_occurrence_target_differences",
                "occurrences": OCCURRENCES,
                "policy_differences": 0,
            },
        )
        return
    # The separate training-input binding is appended only after independent
    # derivative verification via bind_training_inputs().


def bind_training_inputs() -> None:
    from ml.alphazero_lite import verify_seed554_fixed_state_teacher_budget as verifier

    live_generation_preflight()
    preflight()
    verification = verifier.verify(rerun_searches=True)
    write_json(OUT / "derivative-verification.json", verification)
    expected_binding = {
        "schema": "seed554-training-input-binding-v1",
        "label_generation_freeze_sha256": file_sha(
            OUT / "label-generation-freeze.json"
        ),
        "execution_environment_binding_sha256": file_sha(
            OUT / "execution-environment-binding.json"
        ),
        "seed553_recorded_environment_sha256": file_sha(
            ROOT / "docs/data/seed553-fixed-state-budget/registration.json"
        ),
        "source_corrections_sha256": file_sha(OUT / "source-corrections.jsonl"),
        "attempt_ledger_sha256": file_sha(OUT / "attempts.jsonl"),
        "derivative_sha256": {
            f"{arm}/{shard:03d}": file_sha(OUT / arm / f"shard-{shard:03d}.jsonl")
            for arm in ARMS
            for shard in range(SHARDS)
        },
        "schedule_sha256": {
            arm: file_sha(OUT / f"{arm}-training-schedule.json") for arm in ARMS
        },
        "occurrence_mapping_sha256": file_sha(OUT / "occurrence-to-search.jsonl"),
        "search_evidence_sha256": file_sha(OUT / "searches.jsonl"),
        "derivative_verification_sha256": file_sha(
            OUT / "derivative-verification.json"
        ),
        "rows": OCCURRENCES,
        "ordered_identity_sha256": sha(
            "\n".join(
                json.loads(line)["row_id"]
                for line in (OUT / "occurrence-to-search.jsonl")
                .read_text()
                .splitlines()
            ).encode()
        ),
    }
    destination = OUT / "training-input-binding.json"
    if destination.exists():
        if json.loads(destination.read_text()) != expected_binding:
            raise ValueError("conflicting_training_input_binding_preserved")
        return
    write_json(destination, expected_binding)


def reject_existing_training_evidence(output: Path) -> None:
    products = (
        [output / f"{arm}-updates.jsonl" for arm in ARMS]
        + [
            output / f"{arm}-update-{update:03d}.npz"
            for arm in ARMS
            for update in (27, 54, 81, 108)
        ]
        + [
            output / "control-production-equivalence.jsonl",
        ]
        + [
            output / name
            for name in (
                "run-summary.json",
                "development-results.json",
                "development-predictions.jsonl",
            )
        ]
    )
    conflicts = [path.name for path in products if path.exists()]
    if conflicts:
        raise FileExistsError(
            "seed554_training_evidence_already_exists:" + ",".join(conflicts)
        )


def run_seed554_arm(training: Any, arm: str, load_arm: Any) -> dict[str, Any]:
    """Run an arm; the C128 production/Adam gate lives only in seed554."""
    x, p, v, weights, identities = load_arm(arm)
    batches = training.load_schedule(arm)
    arrays = (x, p, v, weights)
    model = training.model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    production_model = training.model() if arm == "control" else None
    production_optimizer = (
        torch.optim.Adam(
            production_model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        if production_model is not None
        else None
    )
    trace_path = OUT / f"{arm}-updates.jsonl"
    gate_path = OUT / "control-production-equivalence.jsonl"
    gate_stream = gate_path.open("x", encoding="utf-8") if arm == "control" else None
    checkpoints: dict[str, str] = {}
    try:
        with trace_path.open("x", encoding="utf-8") as trace:
            for batch in batches:
                indices = batch["row_occurrence_indices"]
                production_parameter_before: str | None = None
                production_optimizer_before: str | None = None
                if production_model is not None and production_optimizer is not None:
                    production_parameter_before = training.reference.state_digest(
                        production_model
                    )
                    production_optimizer_before = training.optimizer_state_id(
                        production_optimizer, production_model
                    )
                record = training.update(model, optimizer, arrays, indices)
                record["update"] = batch["update"]
                record["row_occurrence_ids"] = [identities[index] for index in indices]
                if production_model is not None and production_optimizer is not None:
                    assert gate_stream is not None
                    assert production_parameter_before is not None
                    assert production_optimizer_before is not None
                    index_array = np.asarray(indices, dtype=np.int64)
                    previous_randperm = torch.randperm
                    torch.randperm = lambda n, *a, **kw: torch.arange(
                        n, device=kw.get("device", "cpu")
                    )
                    try:
                        train.train_one_epoch(
                            model=production_model,
                            optimizer=production_optimizer,
                            compact_x=x[index_array],
                            compact_p=p[index_array],
                            compact_v=v[index_array],
                            compact_policy_loss_weights=weights[index_array],
                            replay_indexes=np.arange(len(index_array), dtype=np.int64),
                            batch_size=len(index_array),
                            device=torch.device("cpu"),
                            value_loss_weight=0.5,
                            value_loss="huber",
                            huber_delta=1.0,
                            grad_clip=1.0,
                            epoch=batch["update"],
                            max_optimizer_updates=1,
                        )
                    finally:
                        torch.randperm = previous_randperm
                    production_parameter_after = training.reference.state_digest(
                        production_model
                    )
                    production_optimizer_after = training.optimizer_state_id(
                        production_optimizer, production_model
                    )
                    comparisons = {
                        "parameter_before_matches": production_parameter_before
                        == record["parameter_state_before_sha256"],
                        "parameter_after_matches": production_parameter_after
                        == record["parameter_state_after_sha256"],
                        "optimizer_before_matches": production_optimizer_before
                        == record["optimizer_state_before_sha256"],
                        "optimizer_after_matches": production_optimizer_after
                        == record["optimizer_state_after_sha256"],
                    }
                    record.update(
                        {
                            "production_parameter_state_before_sha256": production_parameter_before,
                            "production_parameter_state_after_sha256": production_parameter_after,
                            "production_optimizer_state_before_sha256": production_optimizer_before,
                            "production_optimizer_state_after_sha256": production_optimizer_after,
                            "production_equivalent": all(comparisons.values()),
                        }
                    )
                    gate_record = {
                        "update": batch["update"],
                        "comparisons": comparisons,
                        "production_parameter_state_before_sha256": production_parameter_before,
                        "production_parameter_state_after_sha256": production_parameter_after,
                        "production_optimizer_state_before_sha256": production_optimizer_before,
                        "production_optimizer_state_after_sha256": production_optimizer_after,
                        "direct_parameter_state_before_sha256": record[
                            "parameter_state_before_sha256"
                        ],
                        "direct_parameter_state_after_sha256": record[
                            "parameter_state_after_sha256"
                        ],
                        "direct_optimizer_state_before_sha256": record[
                            "optimizer_state_before_sha256"
                        ],
                        "direct_optimizer_state_after_sha256": record[
                            "optimizer_state_after_sha256"
                        ],
                        "production_equivalent": record["production_equivalent"],
                    }
                    gate_stream.write(
                        json.dumps(gate_record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    gate_stream.flush()
                    os.fsync(gate_stream.fileno())
                trace.write(json.dumps(record, sort_keys=True) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
                if record.get("production_equivalent") is False:
                    raise RuntimeError(
                        f"seed554_control_production_equivalence_failed:{batch['update']}"
                    )
                if batch["update"] in {27, 54, 81, 108}:
                    checkpoint = OUT / f"{arm}-update-{batch['update']:03d}.npz"
                    np.savez(checkpoint, **train.checkpoint_from_model(model))
                    checkpoints[str(batch["update"])] = training.sha(checkpoint)
    finally:
        if gate_stream is not None:
            gate_stream.close()
    return {
        "arm": arm,
        "rows": len(x),
        "updates": len(batches),
        "exposures": sum(len(batch["row_occurrence_indices"]) for batch in batches),
        "trace_sha256": training.sha(trace_path),
        "checkpoints": checkpoints,
        "production_gate_sha256": training.sha(gate_path) if arm == "control" else None,
    }


def train_arms() -> None:
    from ml.alphazero_lite import seed549_training as training

    training.configure_torch_threads()
    env_binding = json.loads((OUT / "execution-environment-binding.json").read_text())
    runtime = env_binding["runtime"]
    if (sys.version, platform.platform(), np.__version__, torch.__version__) != (
        runtime["python"],
        runtime["platform"],
        runtime["numpy"],
        runtime["torch"],
    ):
        raise ValueError("seed554_training_numerical_runtime_mismatch")
    if (
        torch_threads() != (1, 1)
        or sorted(os.sched_getaffinity(0)) != [31]
        or os.nice(0) != 19
        or {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        != {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    ):
        raise ValueError("seed554_training_resource_or_thread_contract_mismatch")
    limits = training.cgroup_limits()
    if (
        limits["cpu.max"] != "25000 100000"
        or limits["cpu.weight"] != "1"
        or limits["memory.max"] != "4294967296"
    ):
        raise ValueError("seed554_training_cgroup_limit_mismatch")
    preflight()
    binding_path = OUT / "training-input-binding.json"
    if (OUT / "no-op-result.json").exists():
        raise ValueError("seed554_noop_stop_before_training")
    if not binding_path.exists():
        raise ValueError("training_input_binding_missing")
    binding = json.loads(binding_path.read_text())
    if binding.get("label_generation_freeze_sha256") != file_sha(
        OUT / "label-generation-freeze.json"
    ):
        raise ValueError("training_input_binding_freeze_mismatch")
    from ml.alphazero_lite import verify_seed554_fixed_state_teacher_budget as verifier

    verifier.verify()
    reject_existing_training_evidence(OUT)
    # Reuse exact #549 update and production-equivalence implementation while
    # redirecting only its input/output locations and arm names.
    original = training.CORPUS, training.OUT, training.ARMS, training.load_arm

    def load_seed554_arm(
        arm: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if arm not in ARMS:
            raise ValueError(f"unknown_arm:{arm}")
        identity_path = OUT / f"{arm}-training-row-identities.jsonl"
        identities = [
            json.loads(line)["row_id"]
            for line in identity_path.read_text().splitlines()
        ]
        paths = [OUT / arm / f"shard-{shard:03d}.jsonl" for shard in range(SHARDS)]
        x, p, v, _replay, weights = train.load_jsonl_replay(
            paths,
            weights=[1] * SHARDS,
            policy_target_mode="default",
            value_target_mode="default",
            include_policy_loss_weights=True,
        )
        if (
            len(x) != OCCURRENCES
            or len(identities) != OCCURRENCES
            or len(set(identities)) != OCCURRENCES
        ):
            raise ValueError(f"seed554_training_occurrence_mismatch:{arm}")
        return x, p, v, weights, identities

    try:
        training.CORPUS = OUT
        training.OUT = OUT
        training.ARMS = ARMS
        control_summary = run_seed554_arm(training, "control", load_seed554_arm)
        treatment_summary = run_seed554_arm(training, "treatment", load_seed554_arm)
        write_json(
            OUT / "run-summary.json",
            {
                "control": control_summary,
                "treatment": treatment_summary,
                "candidate": "T512 update 108 only",
            },
        )
        # Development labels are the unchanged #549 shared-membership targets.
        training.CORPUS = CORPUS
        training.publish()
    finally:
        training.CORPUS, training.OUT, training.ARMS, training.load_arm = original


def torch_threads() -> tuple[int, int]:
    return torch.get_num_threads(), torch.get_num_interop_threads()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("authenticate", "freeze", "preflight", "generate", "bind", "train"),
    )
    action = parser.parse_args().action
    if action == "authenticate":
        print(json.dumps(authenticate_inputs(), indent=2, sort_keys=True))
    elif action == "freeze":
        freeze()
    elif action == "preflight":
        preflight()
    elif action == "generate":
        generate()
    elif action == "bind":
        bind_training_inputs()
    else:
        train_arms()


if __name__ == "__main__":
    main()
