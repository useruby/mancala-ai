"""Fixed-state paired PUCT budget diagnostic for seed553.

The cohort is selected from seed549 control training occurrences before any
teacher search. Searches use the complete frozen initializer evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
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
RESOURCE_LIMITS = {
    "cpu_quota": "25%",
    "cpu_weight": 1,
    "memory_max_bytes": 4294967296,
    "cpu_affinity": "31",
    "nice": 19,
}
RESOURCE_LAUNCHER = [
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
BUCKETS = ((">32", 33, 48), ("17-32", 17, 32))
BASE_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "visit_count",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_state_id(state: dict[str, Any]) -> str:
    cells = (
        state["player_pits"]
        + state["opponent_pits"]
        + [state["player_store"], state["opponent_store"], state["current_player"]]
    )
    return sha_bytes(json.dumps(cells, separators=(",", ":")).encode())


def encoded_state(state: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        self_play.encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
    )


def encoded_state_id(state: dict[str, Any]) -> str:
    return sha_bytes(encoded_state(state).tobytes())


def _source_candidates(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / CORPUS / "control-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for shard in manifest["shards"]:
        if shard["partition"] != "training":
            continue
        path = root / shard["path"]
        if sha_bytes(path.read_bytes()) != shard["sha256"]:
            raise ValueError(f"seed549_shard_hash_mismatch:{shard['path']}")
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for line_no, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                row = json.loads(line)
                row["_row_number"] = line_no
                grouped[int(row["game_index"])].append(row)
        for game_index, game_rows in grouped.items():
            ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
            if not ordered or ordered[0].get("game_completed") is not True:
                continue
            game = KalahGame.from_state(ordered[0]["game_start_state"])
            moves = ordered[0]["game_moves"]
            for row, move in zip(ordered, moves, strict=True):
                state = game.to_state()
                active = sum(state["player_pits"] + state["opponent_pits"])
                if (
                    active > 16
                    and not game.over()
                    and row.get("game_completed") is True
                ):
                    encoded = encoded_state(state)
                    source_encoded = np.asarray(row["state"], dtype=np.float32)
                    if encoded.tobytes() != source_encoded.tobytes():
                        raise ValueError("seed549_state_decode_reencode_mismatch")
                    candidates.append(
                        {
                            "source": {
                                "path": shard["path"],
                                "sha256": shard["sha256"],
                                "shard": int(shard["shard"]),
                                "game_index": game_index,
                                "move_index": int(row["move_index"]),
                                "row_number": int(row["_row_number"]),
                            },
                            "state": state,
                            "legal_moves": game.possible_moves(),
                            "canonical_state_sha256": canonical_state_id(state),
                            "production_float32_sha256": encoded_state_id(state),
                            "encoded_input_float32": encoded.tolist(),
                            "active_stones": active,
                        }
                    )
                if not game.move(game.pit_index(int(move))):
                    raise ValueError("seed549_control_training_replay_failed")
            if not game.over():
                raise ValueError("seed549_control_training_game_not_terminal")
    return candidates


def build_cohort(root: Path = ROOT) -> list[dict[str, Any]]:
    """Select exactly 128 unique rows in each bucket via seed-label hash rank."""
    root = root.resolve()
    candidates = _source_candidates(root)
    training_rows_path = root / CORPUS / "control-training-row-identities.jsonl"
    occurrence_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for line in training_rows_path.read_text().splitlines():
        if not line.strip():
            continue
        occurrence = json.loads(line)
        source = occurrence["source"]
        key = (
            int(source["shard"]),
            int(source["game_index"]),
            int(source["move_index"]),
        )
        if key in occurrence_index:
            raise ValueError("seed549_training_occurrence_identity_duplicate")
        occurrence_index[key] = occurrence
    selected: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    seen_encoded: set[str] = set()
    for bucket, low, high in BUCKETS:
        members = [row for row in candidates if low <= row["active_stones"] <= high]
        members.sort(
            key=lambda row: (
                sha_bytes(
                    f"seed553:{bucket}:{row['canonical_state_sha256']}:{row['production_float32_sha256']}".encode()
                ),
                row["source"]["path"],
                row["source"]["row_number"],
            )
        )
        taken = 0
        for row in members:
            canonical_id = row["canonical_state_sha256"]
            encoded_id = row["production_float32_sha256"]
            if canonical_id in seen_canonical or encoded_id in seen_encoded:
                continue
            source = row["source"]
            occurrence = occurrence_index.get(
                (source["shard"], source["game_index"], source["move_index"])
            )
            if occurrence is None:
                raise ValueError(
                    "seed553_selected_row_missing_from_training_identities"
                )
            if (
                occurrence["source"]["path"] != source["path"]
                or int(occurrence["source"]["row_number"]) != source["row_number"]
                or occurrence["canonical_state_sha256"] != canonical_id
                or occurrence["production_float32_sha256"] != encoded_id
                or int(occurrence["active_stones"]) != row["active_stones"]
            ):
                raise ValueError("seed553_selected_row_identity_mismatch")
            seen_canonical.add(canonical_id)
            seen_encoded.add(encoded_id)
            selected.append(
                {
                    **row,
                    "source": {**source, "row_id": occurrence["row_id"]},
                    "bucket": bucket,
                    "cohort_index": len(selected),
                }
            )
            taken += 1
            if taken == 128:
                break
        if taken != 128:
            raise ValueError(f"seed553_bucket_shortfall:{bucket}:{taken}")
    return selected


def production_loader_report(
    root: Path, cohort: list[dict[str, Any]]
) -> dict[str, Any]:
    """Authenticate selected float32 inputs through the production loader."""
    from ml.alphazero_lite import train

    groups = {number: [] for number in range(6)}
    for row in cohort:
        groups[int(row["source"]["shard"])].append(row)
    reports = {}
    for number, rows in groups.items():
        path = root / CORPUS / "control" / f"shard-{number:03d}.jsonl"
        loaded = train.load_jsonl(
            path, policy_target_mode="default", value_target_mode="default"
        )
        states = np.asarray(loaded[0], dtype=np.float32)
        for row in rows:
            offset = int(row["source"]["row_number"])
            if (
                states[offset].tobytes()
                != np.asarray(row["encoded_input_float32"], dtype=np.float32).tobytes()
            ):
                raise ValueError("seed553_production_loader_input_mismatch")
        reports[str(number)] = {
            "accepted_rows": int(states.shape[0]),
            "selected_rows_checked": len(rows),
            "loader_shapes": [list(np.asarray(array).shape) for array in loaded],
            "selected_input_identity": "bitwise_equal_production_float32",
        }
    return {
        "loader": "ml.alphazero_lite.train.load_jsonl",
        "source_sha256": sha_bytes((root / "ml/alphazero_lite/train.py").read_bytes()),
        "shards": reports,
    }


def search_config(simulations: int) -> dict[str, Any]:
    return {
        "simulations": simulations,
        "c_puct": 1.25,
        "fpu_mode": "zero",
        "root_fpu_mode": None,
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "visit_count",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
        "dirichlet_alpha": None,
        "dirichlet_epsilon": 0.0,
        "exact_leaf_solve": "disabled",
        "exact_root_solve": "ordinary PUCT entry point; exact-root dispatcher not used",
        "input_encoding": "kalah_v3",
        "search_options": BASE_OPTIONS,
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
    }


class _CountingEvaluator(self_play.Evaluator):
    def __init__(self, inner: Any):
        self.inner = inner
        self.calls = 0

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        self.calls += 1
        return self.inner.evaluate(game)


def run_search(
    state: dict[str, Any], *, simulations: int, seed: int, evaluator: Any | None = None
) -> dict[str, Any]:
    evaluator = evaluator or self_play.CheckpointEvaluator(
        ROOT / INITIALIZER, input_encoding="kalah_v3", cache_size=0
    )
    evaluator = _CountingEvaluator(evaluator)
    game = KalahGame.from_state(state)
    if game.over() or sum(game.pits) <= 16:
        raise ValueError("seed553_search_root_outside_registered_population")
    started = __import__("time").perf_counter()
    visits, root = self_play.PUCT(
        evaluator,
        simulations,
        1.25,
        random.Random(seed),
        fpu_mode="zero",
        root_fpu_mode=None,
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    ).run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    elapsed = __import__("time").perf_counter() - started
    legal = game.possible_moves()
    target = self_play.build_policy_target(visits, legal_moves=legal, temperature=1.0)
    # Highest visit ties resolve to the lowest legal relative action index.
    max_visits = max(int(visits[action]) for action in legal)
    selected = min(action for action in legal if int(visits[action]) == max_visits)
    return {
        "simulations": simulations,
        "seed": seed,
        "visits": [int(value) for value in visits],
        "target": target,
        "selected_move": selected,
        "legal_moves": legal,
        "root_visit_count": int(root.visit_count),
        "root_q_value": float(root.q_value),
        "root_evaluation_value": getattr(root, "evaluation_value", None),
        "root_children": {
            str(action): {
                "prior": float(child.prior),
                "visits": int(child.visit_count),
                "q_value": float(child.q_value),
            }
            for action, child in sorted(root.children.items())
        },
        "evaluator_calls": evaluator.calls,
        "elapsed_seconds": elapsed,
    }


def entropy(target: list[float]) -> float:
    return -sum(value * math.log(value) for value in target if value > 0)


def compare_pair(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    a = np.asarray(low["target"], dtype=np.float32)
    b = np.asarray(high["target"], dtype=np.float32)
    return {
        "action_disagreement": low["selected_move"] != high["selected_move"],
        "total_variation": float(
            0.5 * np.abs(a.astype(np.float64) - b.astype(np.float64)).sum()
        ),
        "bitwise_identical_target": a.tobytes() == b.tobytes(),
        "entropy_128": entropy(low["target"]),
        "entropy_512": entropy(high["target"]),
        "support_128": int(np.count_nonzero(a)),
        "support_512": int(np.count_nonzero(b)),
    }


def classify(disagreements: int, median_tv: float) -> str:
    if disagreements >= 13 or median_tv >= 0.05:
        return "propose_fixed_state_budget_training_ablation"
    return "do_not_expand_teacher_budget_from_this_pilot"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(data)
    pending.replace(path)


def _source_paths(root: Path) -> list[str]:
    # The pilot directly imports self-play, game rules, input encodings and
    # evaluator helpers; hash the complete small experiment code package so
    # transitive local imports are also bound.
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "ml/alphazero_lite").glob("*.py")
    )


def prepare(root: Path = ROOT) -> dict[str, Any]:
    """Create the deterministic cohort and freeze all inputs before search."""
    root = root.resolve()
    out = root / OUT
    if out.exists():
        raise FileExistsError("seed553_evidence_directory_already_exists")
    import torch

    receipt_path = root / CORPUS / "post-generation-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    generation_registration = root / CORPUS / "registration.json"
    if sha_bytes(generation_registration.read_bytes()) != receipt.get(
        "generation_registration_sha256"
    ):
        raise ValueError("seed553_seed549_registration_receipt_mismatch")
    for relative, expected in receipt["products"].items():
        product = root / CORPUS / relative
        if sha_bytes(product.read_bytes()) != expected:
            raise ValueError(f"seed553_seed549_product_receipt_mismatch:{relative}")
    seed549_registration = json.loads(generation_registration.read_text())
    if (
        seed549_registration["components"]["initializer"]["checkpoint_sha256"]
        != INITIALIZER_SHA256
    ):
        raise ValueError("seed553_seed549_initializer_binding_mismatch")

    registration = {
        "schema": "seed553-fixed-state-budget-v1",
        "status": "frozen_before_search",
        "scope": "256 distinct seed549 control-training roots; denoised PUCT target comparison only",
        "interpretation": "target-change and computational-cost pilot; no target-quality or playing-strength claim",
        "source_corpus": {
            "registration": sha_bytes(
                (root / CORPUS / "registration.json").read_bytes()
            ),
            "source_manifest": sha_bytes(
                (root / CORPUS / "control-source-manifest.json").read_bytes()
            ),
            "training_occurrences": sha_bytes(
                (root / CORPUS / "control-training-row-identities.jsonl").read_bytes()
            ),
            "post_generation_receipt": sha_bytes(
                (root / CORPUS / "post-generation-receipt.json").read_bytes()
            ),
            "shards": {
                str(shard): sha_bytes(
                    (
                        root / CORPUS / "control" / f"shard-{shard:03d}.jsonl"
                    ).read_bytes()
                )
                for shard in range(6)
            },
        },
        "initializer": {
            "path": str(INITIALIZER),
            "sha256": sha_bytes((root / INITIALIZER).read_bytes()),
            "expected_sha256": INITIALIZER_SHA256,
            "evaluator_path": "self_play.CheckpointEvaluator(input_encoding=kalah_v3, cache_size=0)",
        },
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines(),
        },
        "source_sha256": {
            path: sha_bytes((root / path).read_bytes()) for path in _source_paths(root)
        },
        "cohort": {
            "size": 256,
            "selection_seed_label": "seed553",
            "buckets": {">32": 128, "17-32": 128},
            "terminal_states_excluded": True,
            "deduplicate_both_identities": True,
        },
        "protocol": {
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
            "search_options": BASE_OPTIONS,
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
        },
        "decision_rule": {
            "if": "at least 13/256 highest-visit actions differ OR median total variation >= 0.05",
            "then": "propose_fixed_state_budget_training_ablation",
            "else": "do_not_expand_teacher_budget_from_this_pilot",
            "interpretation": "planning gate only; does not establish target quality or authorize training",
        },
        "resource_limits": RESOURCE_LIMITS,
        "resource_launcher": RESOURCE_LAUNCHER,
        "historical_seed549_verifier": {
            "status": "failed_in_this_checkout",
            "failure_receipt": "docs/data/seed553-prerequisite-failure.json",
            "executed_dependency_closure": "ordinary PUCT path does not invoke native exact-root solving or tablebase queries; roots are >16 stones and exact-leaf solving is disabled",
            "tablebase_dependency": "outside the seed553 executed dependency closure after the fail-closed ordinary-PUCT regression passes",
        },
    }
    if registration["initializer"]["sha256"] != INITIALIZER_SHA256:
        raise ValueError("seed553_initializer_sha256_mismatch")
    cohort = build_cohort(root)
    loader_report = production_loader_report(root, cohort)
    failure_receipt = json.loads(
        (root / "docs/data/seed553-prerequisite-failure.json").read_text()
    )
    cohort_payload = _json_bytes(cohort)
    input_authentication = {
        "schema": "seed553-input-authentication-v1",
        "status": "authenticated_bounded_inputs",
        "seed549_historical_full_verifier": {
            "status": "failed_in_this_checkout",
            "command": failure_receipt["command"],
            "failing_check": failure_receipt["failing_check"],
            "expected_identity": failure_receipt["expected_identity"],
            "observed_identity": failure_receipt["observed_identity"],
            "receipt_sha256": sha_bytes(
                (root / "docs/data/seed553-prerequisite-failure.json").read_bytes()
            ),
        },
        "seed549": {
            "registration_sha256": registration["source_corpus"]["registration"],
            "post_generation_receipt_sha256": registration["source_corpus"][
                "post_generation_receipt"
            ],
            "source_manifest_sha256": registration["source_corpus"]["source_manifest"],
            "verified_product_bindings": receipt["products"],
            "training_shard_sha256": registration["source_corpus"]["shards"],
            "training_occurrences_sha256": registration["source_corpus"][
                "training_occurrences"
            ],
        },
        "initializer": registration["initializer"],
        "numerical_environment": registration["runtime"],
        "execution_source_sha256": registration["source_sha256"],
        "selected_cohort_sha256": sha_bytes(cohort_payload),
        "production_loader": loader_report,
        "ordinary_puct_dependency_boundary": "ordinary PUCT path does not invoke native exact-root solving or tablebase queries; direct regression patches native loader and both tablebase lookup methods to fail, while confirming ≤16-stone leaves still reach the ordinary evaluator",
        "selected_training_source_rows": [
            {
                "cohort_index": row["cohort_index"],
                "source": row["source"],
                "canonical_state_sha256": row["canonical_state_sha256"],
                "production_float32_sha256": row["production_float32_sha256"],
                "active_stones": row["active_stones"],
            }
            for row in cohort
        ],
    }
    _atomic_write(out / "input-authentication.json", _json_bytes(input_authentication))
    registration["input_authentication_sha256"] = sha_bytes(
        (out / "input-authentication.json").read_bytes()
    )
    registration["cohort_sha256"] = sha_bytes(cohort_payload)
    _atomic_write(out / "cohort.json", _json_bytes(cohort))
    _atomic_write(out / "registration.json", _json_bytes(registration))
    _atomic_write(
        out / "execution-freeze.json",
        _json_bytes(
            {
                "status": "frozen_before_search",
                "registration_sha256": sha_bytes(_json_bytes(registration)),
                "cohort_sha256": registration["cohort_sha256"],
                "input_authentication_sha256": registration[
                    "input_authentication_sha256"
                ],
            }
        ),
    )
    return registration


def preflight(root: Path = ROOT) -> dict[str, Any]:
    """Read-only authentication; rejection occurs before any evidence write/search."""
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    registration = json.loads(reg_path.read_text())
    cohort_path = out / "cohort.json"
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != sha_bytes(reg_path.read_bytes()):
        raise ValueError("seed553_registration_freeze_mismatch")
    if freeze.get("cohort_sha256") != sha_bytes(cohort_path.read_bytes()):
        raise ValueError("seed553_cohort_freeze_mismatch")
    authentication_path = out / "input-authentication.json"
    if freeze.get("input_authentication_sha256") != sha_bytes(
        authentication_path.read_bytes()
    ) or registration.get("input_authentication_sha256") != sha_bytes(
        authentication_path.read_bytes()
    ):
        raise ValueError("seed553_input_authentication_receipt_mismatch")
    cohort = json.loads(cohort_path.read_text())
    if len(cohort) != 256 or sha_bytes(_json_bytes(cohort)) != registration.get(
        "cohort_sha256"
    ):
        raise ValueError("seed553_cohort_registration_mismatch")
    if registration.get("protocol") != {
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
        "search_options": BASE_OPTIONS,
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
    }:
        raise ValueError("seed553_frozen_protocol_semantic_mismatch")
    if registration.get("cohort") != {
        "size": 256,
        "selection_seed_label": "seed553",
        "buckets": {">32": 128, "17-32": 128},
        "terminal_states_excluded": True,
        "deduplicate_both_identities": True,
    }:
        raise ValueError("seed553_frozen_cohort_contract_mismatch")
    expected_decision_rule = {
        "if": "at least 13/256 highest-visit actions differ OR median total variation >= 0.05",
        "then": "propose_fixed_state_budget_training_ablation",
        "else": "do_not_expand_teacher_budget_from_this_pilot",
        "interpretation": "planning gate only; does not establish target quality or authorize training",
    }
    if registration.get("decision_rule") != expected_decision_rule:
        raise ValueError("seed553_frozen_decision_rule_mismatch")
    if (
        registration.get("resource_limits") != RESOURCE_LIMITS
        or registration.get("resource_launcher") != RESOURCE_LAUNCHER
    ):
        raise ValueError("seed553_resource_contract_mismatch")
    historical = registration.get("historical_seed549_verifier", {})
    if (
        historical.get("status") != "failed_in_this_checkout"
        or historical.get("failure_receipt")
        != "docs/data/seed553-prerequisite-failure.json"
    ):
        raise ValueError("seed553_historical_seed549_status_mismatch")
    if _json_bytes(build_cohort(root)) != _json_bytes(cohort):
        raise ValueError("seed553_frozen_cohort_semantic_mismatch")
    if (
        sys.executable != registration["runtime"]["interpreter"]
        or sys.version != registration["runtime"]["python"]
    ):
        raise ValueError("seed553_interpreter_mismatch")
    if (
        np.__version__ != registration["runtime"]["numpy"]
        or platform.platform() != registration["runtime"]["platform"]
    ):
        raise ValueError("seed553_numerical_runtime_mismatch")
    import torch

    if torch.__version__ != registration["runtime"]["torch"]:
        raise ValueError("seed553_torch_runtime_mismatch")
    deps = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if deps != registration["runtime"]["pip_freeze"]:
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
            or correction.get("registration_sha256") != sha_bytes(reg_path.read_bytes())
        ):
            raise ValueError("seed553_verification_correction_binding_mismatch")
        if parent_correction_path is not None and correction.get(
            "parent_correction_sha256"
        ) != sha_bytes(parent_correction_path.read_bytes()):
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
        if sha_bytes((root / relative).read_bytes()) != expected:
            raise ValueError(f"seed553_execution_source_changed:{relative}")
    if sha_bytes((root / INITIALIZER).read_bytes()) != INITIALIZER_SHA256:
        raise ValueError("seed553_initializer_sha256_mismatch")
    frozen_initializer = registration.get("initializer", {})
    if (
        frozen_initializer.get("path") != INITIALIZER.as_posix()
        or frozen_initializer.get("sha256") != INITIALIZER_SHA256
        or frozen_initializer.get("expected_sha256") != INITIALIZER_SHA256
        or frozen_initializer.get("evaluator_path")
        != "self_play.CheckpointEvaluator(input_encoding=kalah_v3, cache_size=0)"
    ):
        raise ValueError("seed553_initializer_registration_mismatch")
    receipt_path = root / CORPUS / "post-generation-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if sha_bytes((root / CORPUS / "registration.json").read_bytes()) != receipt.get(
        "generation_registration_sha256"
    ):
        raise ValueError("seed553_seed549_registration_receipt_mismatch")
    for relative, expected in receipt["products"].items():
        if sha_bytes((root / CORPUS / relative).read_bytes()) != expected:
            raise ValueError(f"seed553_seed549_product_receipt_mismatch:{relative}")
    auth_path = out / "input-authentication.json"
    auth_hash = sha_bytes(auth_path.read_bytes())
    if auth_hash != registration.get("input_authentication_sha256"):
        raise ValueError("seed553_input_authentication_receipt_mismatch")
    auth = json.loads(auth_path.read_text())
    if auth.get("status") != "authenticated_bounded_inputs" or auth.get(
        "selected_cohort_sha256"
    ) != sha_bytes(cohort_path.read_bytes()):
        raise ValueError("seed553_input_authentication_semantic_mismatch")
    if (
        auth.get("seed549_historical_full_verifier", {}).get("status")
        != "failed_in_this_checkout"
    ):
        raise ValueError("seed553_historical_seed549_verification_status_mismatch")
    if (
        auth.get("execution_source_sha256") != registration.get("source_sha256")
        or auth.get("initializer") != registration.get("initializer")
        or auth.get("numerical_environment") != registration.get("runtime")
        or "does not invoke native exact-root solving or tablebase queries"
        not in auth.get("ordinary_puct_dependency_boundary", "")
    ):
        raise ValueError("seed553_input_authentication_binding_mismatch")
    seed549_auth = auth.get("seed549", {})
    manifest_path = root / CORPUS / "control-source-manifest.json"
    identities_path = root / CORPUS / "control-training-row-identities.jsonl"
    expected_shards = {
        str(int(item["shard"])): item["sha256"]
        for item in json.loads(manifest_path.read_text())["shards"]
        if item["partition"] == "training"
    }
    if (
        seed549_auth.get("registration_sha256")
        != sha_bytes((root / CORPUS / "registration.json").read_bytes())
        or seed549_auth.get("post_generation_receipt_sha256")
        != sha_bytes(receipt_path.read_bytes())
        or seed549_auth.get("source_manifest_sha256")
        != sha_bytes(manifest_path.read_bytes())
        or seed549_auth.get("training_occurrences_sha256")
        != sha_bytes(identities_path.read_bytes())
        or seed549_auth.get("verified_product_bindings") != receipt["products"]
        or seed549_auth.get("training_shard_sha256") != expected_shards
    ):
        raise ValueError("seed553_seed549_input_receipt_mismatch")
    expected_selected = [
        {
            "cohort_index": item["cohort_index"],
            "source": item["source"],
            "canonical_state_sha256": item["canonical_state_sha256"],
            "production_float32_sha256": item["production_float32_sha256"],
            "active_stones": item["active_stones"],
        }
        for item in cohort
    ]
    if auth.get("selected_training_source_rows") != expected_selected:
        raise ValueError("seed553_input_authentication_source_rows_mismatch")
    failure_path = root / "docs/data/seed553-prerequisite-failure.json"
    failure = json.loads(failure_path.read_text())
    historical = auth["seed549_historical_full_verifier"]
    if (
        historical.get("command") != failure.get("command")
        or historical.get("failing_check") != failure.get("failing_check")
        or historical.get("expected_identity") != failure.get("expected_identity")
        or historical.get("observed_identity") != failure.get("observed_identity")
        or historical.get("receipt_sha256") != sha_bytes(failure_path.read_bytes())
    ):
        raise ValueError("seed553_historical_verifier_failure_receipt_mismatch")
    if auth.get("production_loader") != production_loader_report(root, cohort):
        raise ValueError("seed553_production_loader_receipt_mismatch")
    return registration


def _verify_resource_scope() -> dict[str, Any]:
    """Fail closed unless currently running in the registered bounded scope."""
    affinity = sorted(os.sched_getaffinity(0))
    nice_value = os.getpriority(os.PRIO_PROCESS, 0)
    cgroup_rel = next(
        (
            line.split(":", 2)[2].strip()
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if cgroup_rel is None:
        raise RuntimeError("seed553_cgroup_v2_scope_missing")
    cgroup = Path("/sys/fs/cgroup") / cgroup_rel.lstrip("/")
    try:
        cpu_quota, cpu_period = (
            int(value) for value in (cgroup / "cpu.max").read_text().split()
        )
        cpu_weight = int((cgroup / "cpu.weight").read_text().strip())
        memory_max = int((cgroup / "memory.max").read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("seed553_resource_scope_unreadable") from error
    if affinity != [31] or nice_value < 19:
        raise RuntimeError("seed553_affinity_or_nice_mismatch")
    if cpu_quota * 4 != cpu_period or cpu_weight != 1 or memory_max != 4294967296:
        raise RuntimeError("seed553_cgroup_resource_limit_mismatch")
    return {
        "cpu_affinity": affinity,
        "nice": nice_value,
        "cpu_quota": f"{cpu_quota}/{cpu_period}",
        "cpu_weight": cpu_weight,
        "memory_max_bytes": memory_max,
        "cgroup": cgroup_rel,
    }


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired rows; kept small and deterministic for publication."""
    if len(records) != 256:
        raise ValueError("seed553_analysis_requires_256_pairs")
    result: dict[str, Any] = {"overall": {}, "by_bucket": {}}

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = [compare_pair(r["searches"]["128"], r["searches"]["512"]) for r in rows]
        disagreements = sum(item["action_disagreement"] for item in pairs)
        median_tv = float(np.median([item["total_variation"] for item in pairs]))
        target_stats = {}
        for budget in ("128", "512"):
            entries = [r["searches"][budget] for r in rows]
            target_stats[budget] = {
                "mean_entropy_nats": float(
                    np.mean([entropy(e["target"]) for e in entries])
                ),
                "mean_support_size": float(
                    np.mean([np.count_nonzero(e["target"]) for e in entries])
                ),
                "total_search_seconds": float(
                    sum(e["elapsed_seconds"] for e in entries)
                ),
                "evaluator_calls": int(sum(e["evaluator_calls"] for e in entries)),
                "configured_simulations": int(sum(e["simulations"] for e in entries)),
                "actual_root_visits": int(sum(e["root_visit_count"] for e in entries)),
            }
        return {
            "n": len(rows),
            "highest_visit_action_disagreements": int(disagreements),
            "median_total_variation": median_tv,
            "mean_total_variation": float(
                np.mean([item["total_variation"] for item in pairs])
            ),
            "bitwise_identical_targets": sum(
                item["bitwise_identical_target"] for item in pairs
            ),
            "target": target_stats,
        }

    result["overall"] = summarize(records)
    for bucket in (">32", "17-32"):
        result["by_bucket"][bucket] = summarize(
            [r for r in records if r["bucket"] == bucket]
        )
    overall = result["overall"]
    result["classification"] = classify(
        overall["highest_visit_action_disagreements"], overall["median_total_variation"]
    )
    result["interpretation"] = (
        "planning gate only; target changes and cost do not establish target quality or playing strength"
    )
    return result


def _record_hash(row: dict[str, Any]) -> str:
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    return sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def execute(root: Path = ROOT) -> dict[str, Any]:
    """Run/resume searches, appending durable per-search evidence and attempts."""
    root = root.resolve()
    preflight(root)
    resource_observation = _verify_resource_scope()
    out = root / OUT
    searches_path = out / "searches.jsonl"
    attempts_path = out / "attempts.jsonl"
    cohort = json.loads((out / "cohort.json").read_text())
    registration_sha = sha_bytes((out / "registration.json").read_bytes())
    cohort_by_index = {int(item["cohort_index"]): item for item in cohort}
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    from ml.alphazero_lite.verify_seed553_fixed_state_budget import (
        validate_search_row,
    )

    if searches_path.exists():
        for line in searches_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("record_sha256") != _record_hash(row):
                raise ValueError("seed553_partial_search_receipt_mismatch")
            key = (int(row["cohort_index"]), int(row["simulations"]))
            if key in completed or key[0] not in range(256) or key[1] not in (128, 512):
                raise ValueError("seed553_conflicting_partial_search_evidence")
            if row.get("registration_sha256") != sha_bytes(
                (out / "registration.json").read_bytes()
            ):
                raise ValueError("seed553_partial_search_registration_mismatch")
            item = cohort_by_index[key[0]]
            validate_search_row(
                row,
                item,
                budget=key[1],
                seed=553000 + key[0],
                registration_sha256=registration_sha,
            )
            completed[key] = row
    prior_attempts = (
        [
            json.loads(line)
            for line in attempts_path.read_text().splitlines()
            if line.strip()
        ]
        if attempts_path.is_file()
        else []
    )
    attempt_counts: dict[tuple[int, int], int] = {}
    first_started_order: list[tuple[int, int]] = []
    pending_start: dict[str, Any] | None = None
    for event in prior_attempts:
        key = (int(event.get("cohort_index", -1)), int(event.get("simulations", -1)))
        if (
            event.get("registration_sha256") != registration_sha
            or key[0] not in cohort_by_index
            or key[1] not in (128, 512)
        ):
            raise ValueError("seed553_partial_attempt_binding_mismatch")
        if event.get("event") == "started":
            if pending_start is not None:
                raise ValueError("seed553_conflicting_partial_attempt_events")
            if key not in attempt_counts:
                pair_index = len(first_started_order) // 2
                first_budget = 128 if pair_index % 2 == 0 else 512
                budget_order = (first_budget, 640 - first_budget)
                expected_key = (
                    pair_index,
                    budget_order[len(first_started_order) % 2],
                )
                if key != expected_key:
                    raise ValueError("seed553_attempt_budget_order_mismatch")
                first_started_order.append(key)
            expected_attempt = attempt_counts.get(key, 0) + 1
            if event.get("attempt_number") != expected_attempt:
                raise ValueError("seed553_attempt_number_mismatch")
            attempt_counts[key] = expected_attempt
            pending_start = event
        elif event.get("event") in {"finished", "failed", "interrupted"}:
            if pending_start is None or any(
                event.get(field) != pending_start.get(field)
                for field in (
                    "cohort_index",
                    "simulations",
                    "seed",
                    "attempt_number",
                    "registration_sha256",
                )
            ):
                raise ValueError("seed553_orphan_attempt_terminal_event")
            if event["event"] == "finished":
                row = completed.get(key)
                if row is None or event.get("record_sha256") != row.get(
                    "record_sha256"
                ):
                    raise ValueError("seed553_finished_attempt_missing_search_record")
            pending_start = None
        else:
            raise ValueError("seed553_unknown_attempt_event")
    for key in completed:
        if attempt_counts.get(key, 0) == 0:
            raise ValueError("seed553_search_record_without_attempt")
    recovery_event = None
    if pending_start is not None:
        key = (int(pending_start["cohort_index"]), int(pending_start["simulations"]))
        existing = completed.get(key)
        recovery_event = {
            **pending_start,
            "event": "finished" if existing else "interrupted",
            "record_sha256": existing["record_sha256"] if existing else None,
            "recovered_at_epoch_ns": time.time_ns(),
        }
        if existing:
            attempt_counts[key] = attempt_counts[key]
    # All partial evidence has been authenticated before this recovery append.
    evaluator = self_play.CheckpointEvaluator(
        root / INITIALIZER, input_encoding="kalah_v3", cache_size=0
    )
    attempts_path.touch(exist_ok=True)
    with (
        attempts_path.open("a", encoding="utf-8") as attempts,
        searches_path.open("a", encoding="utf-8") as evidence,
    ):
        if recovery_event is not None:
            attempts.write(json.dumps(recovery_event, sort_keys=True) + "\n")
            attempts.flush()
            os.fsync(attempts.fileno())
        for item in cohort:
            index = int(item["cohort_index"])
            order = (128, 512) if index % 2 == 0 else (512, 128)
            for budget in order:
                if (index, budget) in completed:
                    continue
                seed = 553000 + index
                started = {
                    "event": "started",
                    "cohort_index": index,
                    "simulations": budget,
                    "seed": seed,
                    "attempt_number": attempt_counts.get((index, budget), 0) + 1,
                    "registration_sha256": sha_bytes(
                        (out / "registration.json").read_bytes()
                    ),
                    "started_at_epoch_ns": time.time_ns(),
                }
                attempts.write(json.dumps(started, sort_keys=True) + "\n")
                attempts.flush()
                os.fsync(attempts.fileno())
                try:
                    result = run_search(
                        item["state"],
                        simulations=budget,
                        seed=seed,
                        evaluator=evaluator,
                    )
                    row = {
                        "cohort_index": index,
                        "bucket": item["bucket"],
                        "canonical_state_sha256": item["canonical_state_sha256"],
                        "production_float32_sha256": item["production_float32_sha256"],
                        "simulations": budget,
                        "seed": seed,
                        "registration_sha256": started["registration_sha256"],
                        "result": result,
                    }
                    row["record_sha256"] = _record_hash(row)
                    evidence.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    evidence.flush()
                    os.fsync(evidence.fileno())
                    completed[(index, budget)] = row
                    terminal = {
                        **started,
                        "event": "finished",
                        "record_sha256": row["record_sha256"],
                        "completed_at_epoch_ns": time.time_ns(),
                    }
                except BaseException as error:
                    terminal = {
                        **started,
                        "event": "failed",
                        "error": repr(error),
                        "completed_at_epoch_ns": time.time_ns(),
                    }
                    attempts.write(json.dumps(terminal, sort_keys=True) + "\n")
                    attempts.flush()
                    os.fsync(attempts.fileno())
                    raise
                attempts.write(json.dumps(terminal, sort_keys=True) + "\n")
                attempts.flush()
                os.fsync(attempts.fileno())
                attempt_counts[(index, budget)] = started["attempt_number"]
    if len(completed) != 512:
        raise ValueError("seed553_search_execution_incomplete")
    paired = []
    for item in cohort:
        index = item["cohort_index"]
        paired.append(
            {
                **item,
                "searches": {
                    str(budget): completed[(index, budget)]["result"]
                    for budget in (128, 512)
                },
            }
        )
    report = analyze_records(paired)
    train_rows = int(
        json.loads((root / CORPUS / "control-source-manifest.json").read_text())[
            "training_rows"
        ]
    )
    report["full_control_training_relabel_cost_estimate_seconds"] = {
        budget: float(
            train_rows
            * report["overall"]["target"][str(budget)]["total_search_seconds"]
            / 256
        )
        for budget in ("128", "512")
    }
    report["full_control_training_population_rows"] = train_rows
    report["resource_observation"] = resource_observation
    _atomic_write(out / "results.json", _json_bytes(report))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "execute"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root)
    elif args.action == "preflight":
        result = preflight(args.root)
    else:
        result = execute(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
