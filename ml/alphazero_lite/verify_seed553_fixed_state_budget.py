"""Independent read-only verifier for the seed553 fixed-state PUCT pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import self_play
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed553-fixed-state-budget")
CORPUS = Path("docs/data/seed549-matched-selfplay")
INITIALIZER = Path("docs/data/seed542-selfplay-4096/initializer.npz")
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
FROZEN_PROTOCOL = {
    "budgets": [128, 512],
    "searches": 512,
    "seed_schedule": "553000 + cohort_index, shared by paired budgets",
    "fresh_tree_rng": True,
    "order": "alternate budget order by cohort index",
    "c_puct": 1.25,
    "fpu_mode": "zero",
    "root_fpu_mode": None,
    "dirichlet_noise": False,
    "reuse_subtree": False,
    "normalize_values": False,
    "tactical_bias": 0.0,
    "exact_leaf_solve": "disabled",
    "exact_root_solve": "ordinary PUCT entry point; exact-root dispatcher not used",
    "input_encoding": "kalah_v3",
    "search_options": {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "visit_count",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
    },
    "remaining_search_options": {
        "value_trust_schedule": None,
        "value_transform": None,
        "evaluator_cache_size": 0,
        "opening_min_simulations": 0,
        "opening_min_simulations_plies": 0,
        "opponent_pool_config": None,
        "opening_cache": None,
        "shadow_checkpoint": None,
        "shadow_q_weight": 0.0,
        "tree_reuse_enabled": False,
    },
    "root_stones": ">16",
    "comparison_target_temperature": 1.0,
    "historical_generation_temperature": "schedule (1.0 through ply 9, 0.1 thereafter); diagnostic targets differ",
    "tie_rule": "lowest legal relative action index",
}
EXPECTED_RESOURCE_LIMITS = {
    "cpu_quota": "25%",
    "cpu_weight": 1,
    "memory_max_bytes": 4294967296,
    "cpu_affinity": "31",
    "nice": 19,
}
EXPECTED_RESOURCE_LAUNCHER = [
    "systemd-run",
    "--user",
    "--scope",
    "--quiet",
    "-p",
    "CPUQuota=25%",
    "-p",
    "CPUWeight=1",
    "-p",
    "MemoryMax=4294967296",
    "--",
    "taskset",
    "-c",
    "31",
    "nice",
    "-n",
    "19",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_identity(state: dict[str, Any]) -> str:
    values = (
        state["player_pits"]
        + state["opponent_pits"]
        + [state["player_store"], state["opponent_store"], state["current_player"]]
    )
    return digest(json.dumps(values, separators=(",", ":")).encode())


def encoded_identity(state: dict[str, Any]) -> tuple[str, np.ndarray]:
    encoded = np.asarray(
        self_play.encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
    )
    return digest(encoded.tobytes()), encoded


def rebuild_cohort(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / CORPUS / "control-source-manifest.json").read_text())
    identities: dict[tuple[int, int, int], dict[str, Any]] = {}
    identity_path = root / CORPUS / "control-training-row-identities.jsonl"
    for line in identity_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source = record["source"]
        identity_key = (
            int(source["shard"]),
            int(source["game_index"]),
            int(source["move_index"]),
        )
        if identity_key in identities:
            raise ValueError("seed553_duplicate_source_occurrence_identity")
        identities[identity_key] = record

    candidates: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        if shard["partition"] != "training":
            continue
        path = root / shard["path"]
        raw = path.read_bytes()
        if digest(raw) != shard["sha256"]:
            raise ValueError(f"seed553_source_shard_hash_mismatch:{shard['shard']}")
        games: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for physical_line, line in enumerate(raw.decode().splitlines()):
            if line.strip():
                source_row = json.loads(line)
                games.setdefault(int(source_row["game_index"]), []).append(
                    (physical_line, source_row)
                )
        for game_index, ordered in games.items():
            ordered.sort(key=lambda pair: int(pair[1]["move_index"]))
            first = ordered[0][1]
            game = KalahGame.from_state(first["game_start_state"])
            moves = first["game_moves"]
            if len(ordered) != len(moves):
                raise ValueError("seed553_source_game_length_mismatch")
            for (line_number, source_row), chosen_move in zip(
                ordered, moves, strict=True
            ):
                state = game.to_state()
                active = sum(game.pits)
                encoded_id, vector = encoded_identity(state)
                if (
                    vector.tobytes()
                    != np.asarray(source_row["state"], dtype=np.float32).tobytes()
                ):
                    raise ValueError("seed553_source_state_encoding_mismatch")
                if active > 16 and not game.over():
                    candidates.append(
                        {
                            "source": {
                                "path": shard["path"],
                                "sha256": shard["sha256"],
                                "shard": int(shard["shard"]),
                                "game_index": game_index,
                                "move_index": int(source_row["move_index"]),
                                "row_number": line_number,
                            },
                            "state": state,
                            "legal_moves": game.possible_moves(),
                            "canonical_state_sha256": canonical_identity(state),
                            "production_float32_sha256": encoded_id,
                            "encoded_input_float32": vector.tolist(),
                            "active_stones": active,
                        }
                    )
                if not game.move(game.pit_index(int(chosen_move))):
                    raise ValueError("seed553_source_trajectory_move_rejected")
            if not game.over():
                raise ValueError("seed553_source_game_not_terminal")

    picked: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    encoded_seen: set[str] = set()
    for bucket, low, high in ((">32", 33, 48), ("17-32", 17, 32)):
        eligible = [row for row in candidates if low <= row["active_stones"] <= high]
        eligible.sort(
            key=lambda row: (
                digest(
                    f"seed553:{bucket}:{row['canonical_state_sha256']}:{row['production_float32_sha256']}".encode()
                ),
                row["source"]["path"],
                row["source"]["row_number"],
            )
        )
        selected_count = 0
        for row in eligible:
            cid = row["canonical_state_sha256"]
            eid = row["production_float32_sha256"]
            if cid in canonical_seen or eid in encoded_seen:
                continue
            src = row["source"]
            occurrence = identities.get(
                (src["shard"], src["game_index"], src["move_index"])
            )
            if occurrence is None:
                raise ValueError("seed553_selected_occurrence_not_registered")
            if (
                occurrence["source"]["path"] != src["path"]
                or int(occurrence["source"]["row_number"]) != src["row_number"]
                or occurrence["canonical_state_sha256"] != cid
                or occurrence["production_float32_sha256"] != eid
                or int(occurrence["active_stones"]) != row["active_stones"]
            ):
                raise ValueError("seed553_selected_occurrence_identity_mismatch")
            canonical_seen.add(cid)
            encoded_seen.add(eid)
            picked.append(
                {
                    **row,
                    "source": {**src, "row_id": occurrence["row_id"]},
                    "bucket": bucket,
                    "cohort_index": len(picked),
                }
            )
            selected_count += 1
            if selected_count == 128:
                break
        if selected_count != 128:
            raise ValueError("seed553_independent_cohort_bucket_shortfall")
    return picked


def validate_cohort(root: Path, archived: list[dict[str, Any]]) -> None:
    if json_bytes(rebuild_cohort(root)) != json_bytes(archived):
        raise ValueError("seed553_independent_cohort_reconstruction_mismatch")


def verify_inputs(
    root: Path, registration: dict[str, Any], cohort: list[dict[str, Any]]
) -> None:
    out = root / OUT
    receipt = json.loads((root / CORPUS / "post-generation-receipt.json").read_text())
    cohort_path = out / "cohort.json"
    registration_path = out / "registration.json"
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != digest(registration_path.read_bytes()):
        raise ValueError("seed553_freeze_registration_mismatch")
    if freeze.get("cohort_sha256") != digest(cohort_path.read_bytes()):
        raise ValueError("seed553_freeze_cohort_mismatch")
    if digest(json_bytes(cohort)) != registration.get("cohort_sha256"):
        raise ValueError("seed553_cohort_registration_hash_mismatch")
    validate_cohort(root, cohort)
    if registration.get("protocol") != FROZEN_PROTOCOL:
        raise ValueError("seed553_frozen_protocol_semantic_mismatch")
    frozen_initializer = registration.get("initializer", {})
    if (
        frozen_initializer.get("path") != INITIALIZER.as_posix()
        or frozen_initializer.get("sha256") != INITIALIZER_SHA256
        or frozen_initializer.get("expected_sha256") != INITIALIZER_SHA256
        or frozen_initializer.get("evaluator_path")
        != "self_play.CheckpointEvaluator(input_encoding=kalah_v3, cache_size=0)"
    ):
        raise ValueError("seed553_frozen_initializer_binding_mismatch")
    historical_registration = registration.get("historical_seed549_verifier", {})
    if (
        historical_registration.get("status") != "failed_in_this_checkout"
        or historical_registration.get("failure_receipt")
        != "docs/data/seed553-prerequisite-failure.json"
        or "does not invoke native exact-root solving or tablebase queries"
        not in historical_registration.get("executed_dependency_closure", "")
    ):
        raise ValueError("seed553_historical_seed549_registration_mismatch")
    expected_decision = {
        "if": "at least 13/256 highest-visit actions differ OR median total variation >= 0.05",
        "then": "propose_fixed_state_budget_training_ablation",
        "else": "do_not_expand_teacher_budget_from_this_pilot",
        "interpretation": "planning gate only; does not establish target quality or authorize training",
    }
    if registration.get("decision_rule") != expected_decision:
        raise ValueError("seed553_frozen_decision_rule_mismatch")
    if registration.get("cohort") != {
        "size": 256,
        "selection_seed_label": "seed553",
        "buckets": {">32": 128, "17-32": 128},
        "terminal_states_excluded": True,
        "deduplicate_both_identities": True,
    }:
        raise ValueError("seed553_frozen_cohort_contract_mismatch")
    if (
        registration.get("resource_limits") != EXPECTED_RESOURCE_LIMITS
        or registration.get("resource_launcher") != EXPECTED_RESOURCE_LAUNCHER
    ):
        raise ValueError("seed553_resource_contract_mismatch")
    auth_path = out / "input-authentication.json"
    auth = json.loads(auth_path.read_text())
    if digest(auth_path.read_bytes()) != registration.get(
        "input_authentication_sha256"
    ):
        raise ValueError("seed553_input_authentication_receipt_hash_mismatch")
    if (
        auth.get("seed549_historical_full_verifier", {}).get("status")
        != "failed_in_this_checkout"
    ):
        raise ValueError("seed553_historical_verifier_status_mismatch")
    if "does not invoke native exact-root solving or tablebase queries" not in auth.get(
        "ordinary_puct_dependency_boundary", ""
    ):
        raise ValueError("seed553_tablebase_dependency_boundary_missing")
    if auth.get("selected_cohort_sha256") != digest(json_bytes(cohort)):
        raise ValueError("seed553_input_receipt_cohort_binding_mismatch")
    selected = [
        {
            "cohort_index": row["cohort_index"],
            "source": row["source"],
            "canonical_state_sha256": row["canonical_state_sha256"],
            "production_float32_sha256": row["production_float32_sha256"],
            "active_stones": row["active_stones"],
        }
        for row in cohort
    ]
    if auth.get("selected_training_source_rows") != selected:
        raise ValueError("seed553_input_receipt_source_rows_mismatch")
    if auth.get("execution_source_sha256") != registration.get("source_sha256"):
        raise ValueError("seed553_input_receipt_execution_source_mismatch")
    if auth.get("initializer") != registration.get("initializer"):
        raise ValueError("seed553_input_receipt_initializer_mismatch")
    if auth.get("numerical_environment") != registration.get("runtime"):
        raise ValueError("seed553_input_receipt_runtime_mismatch")
    seed549 = auth.get("seed549", {})
    seed549_reg = (root / CORPUS / "registration.json").read_bytes()
    seed549_receipt_bytes = (
        root / CORPUS / "post-generation-receipt.json"
    ).read_bytes()
    seed549_manifest_bytes = (
        root / CORPUS / "control-source-manifest.json"
    ).read_bytes()
    seed549_rows_bytes = (
        root / CORPUS / "control-training-row-identities.jsonl"
    ).read_bytes()
    source_manifest = json.loads(seed549_manifest_bytes)
    expected_training_shards = {
        str(int(shard["shard"])): shard["sha256"]
        for shard in source_manifest["shards"]
        if shard["partition"] == "training"
    }
    if (
        seed549.get("registration_sha256") != digest(seed549_reg)
        or seed549.get("post_generation_receipt_sha256")
        != digest(seed549_receipt_bytes)
        or seed549.get("source_manifest_sha256") != digest(seed549_manifest_bytes)
        or seed549.get("training_occurrences_sha256") != digest(seed549_rows_bytes)
        or seed549.get("verified_product_bindings") != receipt["products"]
        or seed549.get("training_shard_sha256") != expected_training_shards
    ):
        raise ValueError("seed553_input_receipt_seed549_binding_mismatch")
    from ml.alphazero_lite import train

    loader_rows = {number: [] for number in range(6)}
    for row in cohort:
        loader_rows[int(row["source"]["shard"])].append(row)
    loader_sha = digest((root / "ml/alphazero_lite/train.py").read_bytes())
    loader_report = {
        "loader": "ml.alphazero_lite.train.load_jsonl",
        "source_sha256": loader_sha,
        "shards": {},
    }
    for number, selected_rows in loader_rows.items():
        shard_path = root / CORPUS / "control" / f"shard-{number:03d}.jsonl"
        arrays = train.load_jsonl(
            shard_path, policy_target_mode="default", value_target_mode="default"
        )
        states = np.asarray(arrays[0], dtype=np.float32)
        for selected_row in selected_rows:
            position = int(selected_row["source"]["row_number"])
            if (
                states[position].tobytes()
                != np.asarray(
                    selected_row["encoded_input_float32"], dtype=np.float32
                ).tobytes()
            ):
                raise ValueError("seed553_production_loader_selected_input_mismatch")
        loader_report["shards"][str(number)] = {
            "accepted_rows": int(states.shape[0]),
            "selected_rows_checked": len(selected_rows),
            "loader_shapes": [list(np.asarray(array).shape) for array in arrays],
            "selected_input_identity": "bitwise_equal_production_float32",
        }
    if auth.get("production_loader") != loader_report:
        raise ValueError("seed553_input_receipt_production_loader_mismatch")
    failure_path = root / "docs/data/seed553-prerequisite-failure.json"
    failure = json.loads(failure_path.read_text())
    historical = auth["seed549_historical_full_verifier"]
    if (
        historical.get("command") != failure.get("command")
        or historical.get("failing_check") != failure.get("failing_check")
        or historical.get("expected_identity") != failure.get("expected_identity")
        or historical.get("observed_identity") != failure.get("observed_identity")
        or historical.get("receipt_sha256") != digest(failure_path.read_bytes())
    ):
        raise ValueError("seed553_historical_verifier_failure_binding_mismatch")
    if (
        digest((root / CORPUS / "registration.json").read_bytes())
        != receipt["generation_registration_sha256"]
    ):
        raise ValueError("seed553_seed549_registration_binding_mismatch")
    for path, expected in receipt["products"].items():
        if digest((root / CORPUS / path).read_bytes()) != expected:
            raise ValueError(f"seed553_seed549_product_binding_mismatch:{path}")
    manifest_path = root / CORPUS / "control-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = {
        int(item["shard"]): item
        for item in manifest["shards"]
        if item["partition"] == "training"
    }
    for number, shard in shards.items():
        if digest((root / shard["path"]).read_bytes()) != shard["sha256"]:
            raise ValueError(f"seed553_control_shard_mismatch:{number}")
    if digest((root / INITIALIZER).read_bytes()) != INITIALIZER_SHA256:
        raise ValueError("seed553_initializer_mismatch")
    runtime = registration["runtime"]
    if sys.executable != runtime["interpreter"] or sys.version != runtime["python"]:
        raise ValueError("seed553_interpreter_mismatch")
    if np.__version__ != runtime["numpy"] or platform.platform() != runtime["platform"]:
        raise ValueError("seed553_numerical_runtime_mismatch")
    import torch

    if torch.__version__ != runtime["torch"]:
        raise ValueError("seed553_torch_runtime_mismatch")
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if installed != runtime["pip_freeze"]:
        raise ValueError("seed553_dependency_closure_mismatch")
    expected_sources = dict(registration["source_sha256"])
    correction_paths = [out / "verification-correction.json"]
    correction_number = 2
    while (out / f"verification-correction-{correction_number:02d}.json").is_file():
        correction_paths.append(
            out / f"verification-correction-{correction_number:02d}.json"
        )
        correction_number += 1
    parent_correction_path: Path | None = None
    for correction_path in correction_paths:
        if not correction_path.is_file():
            raise ValueError("seed553_verification_correction_chain_gap")
        correction = json.loads(correction_path.read_text())
        if (
            correction.get("schema") != "seed553-verification-correction-v1"
            or correction.get("status") != "append_only_verification_correction"
            or correction.get("registration_sha256")
            != digest(registration_path.read_bytes())
        ):
            raise ValueError("seed553_verification_correction_binding_mismatch")
        if parent_correction_path is not None and correction.get(
            "parent_correction_sha256"
        ) != digest(parent_correction_path.read_bytes()):
            raise ValueError("seed553_verification_correction_chain_mismatch")
        allowed = {
            "ml/alphazero_lite/seed553_fixed_state_budget.py",
            "ml/alphazero_lite/test_seed553_fixed_state_budget.py",
            "ml/alphazero_lite/verify_seed553_fixed_state_budget.py",
        }
        correction_sources = set(correction.get("source_corrections", {}))
        if not correction_sources or not correction_sources.issubset(allowed):
            raise ValueError("seed553_verification_correction_source_set_mismatch")
        for relative, correction_item in correction["source_corrections"].items():
            if correction_item.get("frozen_sha256") != expected_sources.get(relative):
                raise ValueError(
                    f"seed553_verification_correction_parent_mismatch:{relative}"
                )
            expected_sources[relative] = correction_item["corrected_sha256"]
        parent_correction_path = correction_path
    for relative, expected in expected_sources.items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError(f"seed553_source_identity_mismatch:{relative}")
    seen_canonical: set[str] = set()
    seen_encoded: set[str] = set()
    bucket_counts = {">32": 0, "17-32": 0}
    for index, item in enumerate(cohort):
        state = item["state"]
        game = KalahGame.from_state(state)
        stones = sum(game.pits)
        cid = canonical_identity(state)
        eid, encoded = encoded_identity(state)
        if stones <= 16 or item["active_stones"] != stones:
            raise ValueError("seed553_invalid_root_stone_count")
        if item["cohort_index"] != index or item["canonical_state_sha256"] != cid:
            raise ValueError("seed553_cohort_canonical_identity_mismatch")
        if item["production_float32_sha256"] != eid:
            raise ValueError("seed553_cohort_encoded_identity_mismatch")
        if (
            encoded.tobytes()
            != np.asarray(item["encoded_input_float32"], dtype=np.float32).tobytes()
        ):
            raise ValueError("seed553_cohort_encoding_bytes_mismatch")
        if game.over() or game.possible_moves() != item["legal_moves"]:
            raise ValueError("seed553_cohort_legal_state_mismatch")
        if cid in seen_canonical or eid in seen_encoded:
            raise ValueError("seed553_cohort_duplicate_identity")
        seen_canonical.add(cid)
        seen_encoded.add(eid)
        expected_bucket = ">32" if stones > 32 else "17-32"
        if item["bucket"] != expected_bucket:
            raise ValueError("seed553_cohort_bucket_mismatch")
        bucket_counts[expected_bucket] += 1
    if bucket_counts != {">32": 128, "17-32": 128}:
        raise ValueError("seed553_cohort_bucket_count_mismatch")


class _CallCounter(self_play.Evaluator):
    def __init__(self, evaluator: Any):
        self.inner = evaluator
        self.calls = 0

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        self.calls += 1
        return self.inner.evaluate(game)


def independent_search(
    state: dict[str, Any], budget: int, seed: int, evaluator: Any
) -> dict[str, Any]:
    """Recompute a search directly; deliberately does not call publisher runner helpers."""
    game = KalahGame.from_state(state)
    if game.over() or sum(game.pits) <= 16:
        raise ValueError("seed553_recompute_root_invalid")
    counted = _CallCounter(evaluator)
    visits, root = self_play.PUCT(
        counted,
        budget,
        1.25,
        __import__("random").Random(seed),
        fpu_mode="zero",
        root_fpu_mode=None,
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    ).run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    legal = game.possible_moves()
    target = np.asarray(
        self_play.build_policy_target(visits, legal_moves=legal, temperature=1.0),
        dtype=np.float32,
    )
    highest = max(int(visits[action]) for action in legal)
    action = min(move for move in legal if int(visits[move]) == highest)
    return {
        "simulations": budget,
        "seed": seed,
        "visits": [int(value) for value in visits],
        "target": target.tolist(),
        "selected_move": action,
        "legal_moves": legal,
        "root_visit_count": int(root.visit_count),
        "root_q_value": float(root.q_value),
        "root_evaluation_value": getattr(root, "evaluation_value", None),
        "root_children": {
            str(move): {
                "prior": float(child.prior),
                "visits": int(child.visit_count),
                "q_value": float(child.q_value),
            }
            for move, child in sorted(root.children.items())
        },
        "evaluator_calls": counted.calls,
    }


def _search_matches(archived: dict[str, Any], recomputed: dict[str, Any]) -> bool:
    for key in recomputed:
        if key == "target":
            if (
                np.asarray(archived.get(key), dtype=np.float32).tobytes()
                != np.asarray(recomputed[key], dtype=np.float32).tobytes()
            ):
                return False
        elif archived.get(key) != recomputed[key]:
            return False
    seconds = archived.get("elapsed_seconds")
    return isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds > 0


def validate_search_row(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    budget: int,
    seed: int,
    registration_sha256: str,
) -> None:
    """Check row semantics after its ordinary row receipt has been rebound."""
    if record.get("record_sha256") != digest(
        compact({key: value for key, value in record.items() if key != "record_sha256"})
    ):
        raise ValueError("seed553_search_row_receipt_mismatch")
    if (
        record.get("cohort_index") != item["cohort_index"]
        or record.get("bucket") != item["bucket"]
        or record.get("canonical_state_sha256") != item["canonical_state_sha256"]
        or record.get("production_float32_sha256") != item["production_float32_sha256"]
        or record.get("simulations") != budget
        or record.get("seed") != seed
        or record.get("registration_sha256") != registration_sha256
    ):
        raise ValueError("seed553_search_row_binding_mismatch")
    result = record.get("result")
    if (
        not isinstance(result, dict)
        or result.get("simulations") != budget
        or result.get("seed") != seed
    ):
        raise ValueError("seed553_search_budget_or_seed_mismatch")
    legal = item["legal_moves"]
    if result.get("legal_moves") != legal:
        raise ValueError("seed553_search_legal_action_mapping_mismatch")
    raw_visits = result.get("visits")
    if (
        not isinstance(raw_visits, list)
        or len(raw_visits) != 6
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_visits
        )
    ):
        raise ValueError("seed553_search_visits_invalid")
    visits = np.asarray(raw_visits, dtype=np.int64)
    if np.any(visits < 0) or int(visits.sum()) != budget:
        raise ValueError("seed553_search_visits_invalid")
    if any(visits[action] != 0 for action in range(6) if action not in legal):
        raise ValueError("seed553_illegal_action_visit")
    expected_target = np.asarray(
        self_play.build_policy_target(visits, legal_moves=legal, temperature=1.0),
        dtype=np.float32,
    )
    actual_target = np.asarray(result.get("target"), dtype=np.float32)
    if (
        actual_target.shape != (6,)
        or actual_target.tobytes() != expected_target.tobytes()
    ):
        raise ValueError("seed553_target_does_not_match_visits")
    highest = int(max(visits[action] for action in legal))
    selected = min(action for action in legal if visits[action] == highest)
    if result.get("selected_move") != selected:
        raise ValueError("seed553_selected_action_tie_rule_mismatch")
    if result.get("root_visit_count") != budget:
        raise ValueError("seed553_root_visit_count_mismatch")
    root_q = result.get("root_q_value")
    if not isinstance(root_q, (int, float)) or not math.isfinite(root_q):
        raise ValueError("seed553_root_q_statistic_invalid")
    if result.get("root_evaluation_value") is not None:
        raise ValueError("seed553_root_evaluation_statistic_invalid")
    children = result.get("root_children")
    if not isinstance(children, dict) or set(children) != {
        str(action) for action in legal
    }:
        raise ValueError("seed553_root_child_mapping_mismatch")
    prior_sum = 0.0
    for action in legal:
        child = children[str(action)]
        prior = child.get("prior")
        q_value = child.get("q_value")
        if (
            child.get("visits") != int(visits[action])
            or not isinstance(prior, (int, float))
            or not math.isfinite(prior)
            or prior < 0
            or not isinstance(q_value, (int, float))
            or not math.isfinite(q_value)
        ):
            raise ValueError("seed553_root_child_statistics_mismatch")
        prior_sum += prior
    if not math.isclose(prior_sum, 1.0, abs_tol=1e-5):
        raise ValueError("seed553_root_prior_normalization_mismatch")
    elapsed = result.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("seed553_search_timing_invalid")
    if (
        not isinstance(result.get("evaluator_calls"), int)
        or result["evaluator_calls"] < 1
    ):
        raise ValueError("seed553_evaluator_accounting_invalid")


def _mean_entropy(target: list[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in target if p > 0)


def _reconstruct_report(
    paired: list[dict[str, Any]], training_rows: int
) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        tvs = []
        disagreements = 0
        identical = 0
        for row in rows:
            small = row["searches"]["128"]
            large = row["searches"]["512"]
            a = np.asarray(small["target"], dtype=np.float64)
            b = np.asarray(large["target"], dtype=np.float64)
            tvs.append(float(0.5 * np.abs(a - b).sum()))
            disagreements += int(small["selected_move"] != large["selected_move"])
            identical += int(
                np.asarray(small["target"], dtype=np.float32).tobytes()
                == np.asarray(large["target"], dtype=np.float32).tobytes()
            )
        stats = {}
        for budget in ("128", "512"):
            entries = [row["searches"][budget] for row in rows]
            stats[budget] = {
                "mean_entropy_nats": float(
                    np.mean([_mean_entropy(x["target"]) for x in entries])
                ),
                "mean_support_size": float(
                    np.mean([np.count_nonzero(x["target"]) for x in entries])
                ),
                "total_search_seconds": float(
                    sum(x["elapsed_seconds"] for x in entries)
                ),
                "evaluator_calls": sum(x["evaluator_calls"] for x in entries),
                "configured_simulations": sum(x["simulations"] for x in entries),
                "actual_root_visits": sum(x["root_visit_count"] for x in entries),
            }
        median = float(np.median(tvs))
        return {
            "n": len(rows),
            "highest_visit_action_disagreements": disagreements,
            "median_total_variation": median,
            "mean_total_variation": float(np.mean(tvs)),
            "bitwise_identical_targets": identical,
            "target": stats,
        }

    overall = summarize(paired)
    buckets = {
        key: summarize([r for r in paired if r["bucket"] == key])
        for key in (">32", "17-32")
    }
    classification = (
        "propose_fixed_state_budget_training_ablation"
        if overall["highest_visit_action_disagreements"] >= 13
        or overall["median_total_variation"] >= 0.05
        else "do_not_expand_teacher_budget_from_this_pilot"
    )
    return {
        "overall": overall,
        "by_bucket": buckets,
        "classification": classification,
        "interpretation": "planning gate only; target changes and cost do not establish target quality or playing strength",
        "full_control_training_relabel_cost_estimate_seconds": {
            budget: float(
                training_rows * overall["target"][budget]["total_search_seconds"] / 256
            )
            for budget in ("128", "512")
        },
        "full_control_training_population_rows": training_rows,
    }


def validate_published_report(
    published: dict[str, Any], paired: list[dict[str, Any]], training_rows: int
) -> None:
    expected = _reconstruct_report(paired, training_rows)
    comparable = {
        key: value for key, value in published.items() if key != "resource_observation"
    }
    if json_bytes(expected) != json_bytes(comparable):
        raise ValueError("seed553_published_aggregate_mismatch")


def verify_attempt_ledger(
    path: Path,
    *,
    registration_sha256: str,
    completed: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, int]:
    attempts: dict[tuple[int, int], int] = {}
    pending: dict[str, Any] | None = None
    finished_keys: set[tuple[int, int]] = set()
    first_started_order: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        key = (int(event.get("cohort_index", -1)), int(event.get("simulations", -1)))
        if (
            event.get("registration_sha256") != registration_sha256
            or event.get("seed") != 553000 + key[0]
            or key[0] not in range(256)
            or key[1] not in (128, 512)
        ):
            raise ValueError("seed553_attempt_binding_mismatch")
        if event.get("event") == "started":
            if pending is not None:
                raise ValueError("seed553_attempt_start_without_terminal")
            if key not in attempts:
                pair_index = len(first_started_order) // 2
                first_budget = 128 if pair_index % 2 == 0 else 512
                expected_key = (
                    pair_index,
                    (first_budget, 512 if first_budget == 128 else 128)[
                        len(first_started_order) % 2
                    ],
                )
                if key != expected_key:
                    raise ValueError("seed553_attempt_budget_order_mismatch")
                first_started_order.append(key)
            count = attempts.get(key, 0) + 1
            if event.get("attempt_number") != count:
                raise ValueError("seed553_attempt_ordinal_mismatch")
            attempts[key] = count
            pending = event
            continue
        if (
            event.get("event") not in {"finished", "failed", "interrupted"}
            or pending is None
        ):
            raise ValueError("seed553_attempt_terminal_without_start")
        if any(
            event.get(key_name) != pending.get(key_name)
            for key_name in (
                "cohort_index",
                "simulations",
                "seed",
                "attempt_number",
                "registration_sha256",
            )
        ):
            raise ValueError("seed553_attempt_pair_mismatch")
        if event["event"] == "finished":
            row = completed.get(key)
            if (
                row is None
                or event.get("record_sha256") != row.get("record_sha256")
                or key in finished_keys
            ):
                raise ValueError("seed553_attempt_finish_receipt_mismatch")
            finished_keys.add(key)
        pending = None
    if pending is not None:
        raise ValueError("seed553_attempt_ledger_ends_mid_search")
    if finished_keys != set(completed):
        raise ValueError("seed553_completed_search_attempt_coverage_mismatch")
    expected_order = [
        (index, budget)
        for index in range(256)
        for budget in ((128, 512) if index % 2 == 0 else (512, 128))
    ]
    if first_started_order != expected_order:
        raise ValueError("seed553_attempt_budget_order_incomplete")
    return {
        "attempts": sum(attempts.values()),
        "retried_searches": sum(count > 1 for count in attempts.values()),
    }


def verify(root: Path = ROOT, *, recompute: bool = True) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    tracked = {root / relative for relative in reg["source_sha256"]}
    tracked.update(path for path in out.rglob("*") if path.is_file())
    tracked.update(
        root / path
        for path in (
            INITIALIZER,
            CORPUS / "registration.json",
            CORPUS / "post-generation-receipt.json",
            CORPUS / "control-source-manifest.json",
            CORPUS / "control-training-row-identities.jsonl",
            Path("docs/data/seed553-prerequisite-failure.json"),
        )
    )
    tracked.update(
        root / CORPUS / "control" / f"shard-{index:03d}.jsonl" for index in range(6)
    )
    before = {
        path: (digest(path.read_bytes()), path.stat().st_mtime_ns) for path in tracked
    }
    cohort = json.loads((out / "cohort.json").read_text())
    verify_inputs(root, reg, cohort)
    rows = [
        json.loads(line)
        for line in (out / "searches.jsonl").read_text().splitlines()
        if line.strip()
    ]
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    index_map = {int(item["cohort_index"]): item for item in cohort}
    registration_sha = digest((out / "registration.json").read_bytes())
    for row in rows:
        if row.get("record_sha256") != digest(
            compact({k: v for k, v in row.items() if k != "record_sha256"})
        ):
            raise ValueError("seed553_search_receipt_hash_mismatch")
        index = int(row["cohort_index"])
        budget = int(row["simulations"])
        item = index_map.get(index)
        if item is None or budget not in (128, 512) or (index, budget) in indexed:
            raise ValueError("seed553_search_index_budget_conflict")
        if (
            row.get("seed") != 553000 + index
            or row.get("registration_sha256") != registration_sha
        ):
            raise ValueError("seed553_search_seed_registration_mismatch")
        validate_search_row(
            row,
            item,
            budget=budget,
            seed=553000 + index,
            registration_sha256=registration_sha,
        )
        indexed[(index, budget)] = row
    if len(indexed) != 512:
        raise ValueError("seed553_search_evidence_incomplete")
    attempts = verify_attempt_ledger(
        out / "attempts.jsonl", registration_sha256=registration_sha, completed=indexed
    )
    evaluator = self_play.CheckpointEvaluator(
        root / INITIALIZER, input_encoding="kalah_v3", cache_size=0
    )
    paired = []
    for item in cohort:
        index = int(item["cohort_index"])
        pair = {}
        for budget in (128, 512):
            archived = indexed[(index, budget)]["result"]
            if recompute:
                rebuilt = independent_search(
                    item["state"], budget, 553000 + index, evaluator
                )
                if not _search_matches(archived, rebuilt):
                    raise ValueError(
                        f"seed553_recomputed_search_mismatch:{index}:{budget}"
                    )
            pair[str(budget)] = archived
        paired.append({**item, "searches": pair})
    training_rows = int(
        json.loads((root / CORPUS / "control-source-manifest.json").read_text())[
            "training_rows"
        ]
    )
    published = json.loads((out / "results.json").read_text())
    rebuilt_report = _reconstruct_report(paired, training_rows)
    resource = published.get("resource_observation")
    if (
        not isinstance(resource, dict)
        or resource.get("cpu_affinity") != [31]
        or int(resource.get("nice", -1)) < 19
        or int(resource.get("cpu_weight", -1)) != 1
        or int(resource.get("memory_max_bytes", -1)) != 4294967296
        or not isinstance(resource.get("cgroup"), str)
    ):
        raise ValueError("seed553_resource_observation_invalid")
    rebuilt_report["resource_observation"] = resource
    validate_published_report(published, paired, training_rows)
    after = {
        path: (digest(path.read_bytes()), path.stat().st_mtime_ns) for path in tracked
    }
    if before != after:
        raise RuntimeError("seed553_readonly_verification_changed_evidence")
    return {
        "status": "verified",
        "searches": 512,
        "cohort": 256,
        "classification": rebuilt_report["classification"],
        "attempt_accounting": attempts,
        "recomputed": recompute,
        "evidence_files_checked": len(tracked),
        "evidence_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
