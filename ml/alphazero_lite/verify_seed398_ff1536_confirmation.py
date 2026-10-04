"""Read-only verifier for the exploratory FF1536 publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed398_ff1536_analysis as analysis
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed, stable_hash
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-ff1536-confirmation"
PAIRED = ROOT / "docs/data/seed398-paired-first-action"
E4 = ROOT / "docs/data/seed398-e4-reference-diagnostic"
PERSISTENCE = ROOT / "docs/data/seed398-search-budget-persistence"
SUITE_SHA256 = "10dcb48c9a6bc268ab45ec0a67895b335bb9b669b309115d6ff37b6218131423"
REFERENCES = {
    "seed455": PAIRED / "native-outcomes.jsonl",
    "original_O0_E4": E4 / "outcomes.jsonl",
}
SOURCE_PATHS = {
    "runner": ROOT / "ml/alphazero_lite/seed398_ff1536_confirmation.py",
    "paired_runner": ROOT / "ml/alphazero_lite/seed398_paired_first_action.py",
    "arena": ROOT / "ml/alphazero_lite/arena.py",
    "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
    "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
    "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
    "exact_root_decision": ROOT / "ml/alphazero_lite/exact_root_decision.py",
    "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    paths = {
        "registration": DATA / "registration.json",
        "new_outcomes": DATA / "new-outcomes.jsonl",
        "analysis": DATA / "analysis.json",
        "matrix": DATA / "provenance-matrix.json",
        "results": DATA / "results.md",
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    binding = _load(DATA / "publication-binding.json")
    correction = _load(DATA / "verification-correction-receipt.json")
    binding_bytes = (DATA / "publication-binding.json").read_bytes()
    require(
        correction["schema"]
        == "seed398-ff1536-confirmation-verification-correction-v1",
        "correction_schema",
    )
    require(
        correction["original_publication_binding_sha256"]
        == hashlib.sha256(binding_bytes).hexdigest(),
        "correction_original_binding",
    )
    require(
        correction["historical_verifier_sha256"] == binding["verifier_sha256"],
        "correction_historical_verifier",
    )
    require(
        correction["current_verifier_sha256"] == sha(Path(__file__)),
        "correction_current_verifier",
    )
    require(
        correction["verification_helpers"]["analysis_module_sha256"]
        == sha(ROOT / "ml/alphazero_lite/seed398_ff1536_analysis.py"),
        "correction_analysis_helper",
    )
    require(
        correction["verification_helpers"]["evaluation_seed_contract_sha256"]
        == sha(ROOT / "ml/alphazero_lite/evaluation_seed_contract.py"),
        "correction_seed_helper",
    )
    require(
        correction["verification_helpers"]["pure_rules_sha256"]
        == sha(ROOT / "ml/alphazero_lite/kalah_rules.py"),
        "correction_rules_helper",
    )
    reg = json.loads(before["registration"])
    require(
        reg["schema"] == "seed398-ff1536-confirmation-registration-v1",
        "registration_schema",
    )
    require(
        reg["source_sha256"] == {k: sha(v) for k, v in SOURCE_PATHS.items()},
        "execution_source_binding",
    )
    paired_reg = _load(PAIRED / "registration.json")
    e4_reg = _load(E4 / "registration.json")
    persistence_reg = _load(PERSISTENCE / "registration.json")
    require(
        sha(PAIRED / "registration.json") == reg["state_registration_sha256"],
        "state_registration_binding",
    )
    require(len(reg["probe_sha256"]) == 64, "frozen_probe_binding")
    require(
        len(persistence_reg["publication_binding_sha256"]) == 64,
        "persistence_publication_binding",
    )
    states = {s["opening_index"]: s for s in paired_reg["states"]}
    expected_actions = {
        r["opening_index"]: r["ff1536_action"] for r in reg["action_mapping"]
    }
    require(
        reg["runtime_identities"] == _parent_runtime_identities(paired_reg, e4_reg),
        "runtime_identity_binding",
    )
    require(
        reg["persistence_registration_sha256"]
        == sha(PERSISTENCE / "registration.json"),
        "persistence_registration_binding",
    )
    require(
        persistence_reg["schema"]
        == "seed398-search-budget-persistence-registration-v1",
        "persistence_registration_schema",
    )
    require(len(reg["action_mapping"]) == 64, "action_mapping_count")
    for row in reg["action_mapping"]:
        require(
            row["ff1536_action"] == expected_actions[row["opening_index"]],
            "action_mapping_identity",
        )
    new_expected = [(372, 4), (260, 1), (164, 2), (217, 0)]
    require(
        [
            (row["opening_index"], row["forced_action"])
            for row in reg["new_cases"]
            if row["reference"] == "seed455"
        ]
        == new_expected,
        "new_case_registration",
    )
    for reference in ("seed455", "original_O0_E4"):
        counts = {
            action: sum(
                row["reference"] == reference
                and row["source"] == "reused"
                and row["source_case_identity"]["action"] == action
                for row in reg["reuse_plan"]
            )
            for action in ("FF", "SS")
        }
        require(counts == {"FF": 40, "SS": 20}, f"reuse_action_accounting:{reference}")

    new_rows = [
        json.loads(line)
        for line in before["new_outcomes"].decode().splitlines()
        if line
    ]
    require(len(new_rows) == 8, "new_outcome_count")
    new_case_keys = set()
    for wrapped in new_rows:
        key = wrapped["opening_index"], wrapped["reference"]
        require(key not in new_case_keys, "duplicate_new_outcome")
        new_case_keys.add(key)
        case = next(
            item
            for item in reg["new_cases"]
            if (item["opening_index"], item["reference"]) == key
        )
        require(case["forced_action"] == wrapped["forced_action"], "new_forced_action")
        state = dict(states[key[0]])
        state["ff_action_384"] = case["forced_action"]
        _validate_outcome(wrapped["outcome"], state)
        _verify_seeds(wrapped["outcome"], state)

    ledgers = {}
    for reference, path in REFERENCES.items():
        expected_hash = next(
            item["source_ledger_sha256"]
            for item in reg["reuse_plan"]
            if item["reference"] == reference and item["source"] == "reused"
        )
        require(sha(path) == expected_hash, f"source_ledger_hash:{reference}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        indexed = {}
        for row in rows:
            key = row["opening_index"], row["action"], row["budget"]
            require(key not in indexed, f"duplicate_source_outcome:{reference}")
            _validate_outcome(row, states[row["opening_index"]])
            _verify_seeds(row, states[row["opening_index"]])
            indexed[key] = row
        require(len(indexed) == 256, f"source_ledger_count:{reference}")
        ledgers[reference] = rows
    for item in reg["reuse_plan"]:
        if item["source"] == "reused":
            source_key = item["source_case_identity"]
            source_row = next(
                row
                for row in ledgers[item["reference"]]
                if (row["opening_index"], row["action"], row["budget"])
                == (
                    source_key["opening_index"],
                    source_key["action"],
                    source_key["budget"],
                )
            )
            require(source_row == item["outcome"], "reused_outcome_source_mismatch")

    report, matrix = analysis.calculate(
        reg, new_rows, ledgers["seed455"], ledgers["original_O0_E4"]
    )
    require(len(matrix) == 128, "provenance_matrix_count")
    require(json.loads(before["analysis"]) == report, "analysis_reconciliation")
    require(json.loads(before["matrix"]) == matrix, "matrix_reconciliation")
    require(
        binding["hashes"]
        == {name: hashlib.sha256(data).hexdigest() for name, data in before.items()},
        "publication_hash_binding",
    )
    require(
        binding.get("analysis_sha256")
        == sha(ROOT / "ml/alphazero_lite/seed398_ff1536_analysis.py"),
        "analysis_module_hash_binding",
    )
    after = {name: path.read_bytes() for name, path in paths.items()}
    require(before == after, "verifier_mutated_publication")
    return {
        "status": "verified",
        "new_continuations": 8,
        "reused_outcomes": 120,
        "provenance_cases": len(matrix),
        "analysis_sha256": sha(paths["analysis"]),
        "primary_mean_gain": report["primary"]["mean_gain"],
        "bootstrap_95_interval": report["primary"]["bootstrap_95_interval"],
        "reference_mean_gains": report["reference_mean_gains"],
    }


def _verify_seeds(row: dict[str, Any], state_row: dict[str, Any]) -> None:
    game = KalahGame.from_state(state_row["state"])
    forced = (
        state_row["ss_action_384"]
        if row["action"] == "SS"
        else state_row["ff_action_384"]
    )
    if forced not in game.possible_moves() or not game.move(game.pit_index(forced)):
        raise ValueError("forced_action_illegal")
    for ply, move in enumerate(row["trajectory"][1:], start=1):
        state_hash = stable_hash(game.to_state())
        context = {
            "contract_version": "azlite_eval_seed_v2",
            "base_seed": 406,
            "suite_sha256": SUITE_SHA256,
            "opening_index": state_row["opening_index"],
            "opening_state_hash": state_row["state_hash"],
            "challenger_player": row["root_player"],
            "game_within_opening": 0,
            "ply": ply,
            "canonical_current_state_hash": state_hash,
            "acting_role": "challenger"
            if game.current_player == row["root_player"]
            else "current",
        }
        seed, context_hash = derive_search_seed(**context)
        require(
            move["seed"] == seed and move["seed_context_hash"] == context_hash,
            "seed_context_mismatch",
        )
        if not game.move(game.pit_index(move["action_relative"])):
            raise ValueError("trajectory_move_illegal")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _parent_runtime_identities(
    paired: dict[str, Any], e4: dict[str, Any]
) -> dict[str, Any]:
    refs = {"seed455": paired["reference"], "original_O0_E4": e4["reference"]}
    return {
        "references": {
            name: {
                "artifact_files": ref["artifact_file_sha256"],
                "checkpoint_sha256": ref["checkpoint_sha256"],
            }
            for name, ref in refs.items()
        },
        "native_sha256": paired["reference"]["native_probe_sha256"],
        "tablebase_sha256": paired["reference"]["tablebase_sha256"],
    }


def _validate_outcome(row: dict[str, Any], state: dict[str, Any]) -> None:
    require(row["opening_index"] == state["opening_index"], "outcome_opening_identity")
    require(row["state_hash"] == state["state_hash"], "outcome_state_identity")
    game = KalahGame.from_state(state["state"])
    root = game.current_player
    trajectory = row["trajectory"]
    require(bool(trajectory), "missing_trajectory")
    first = trajectory[0]
    action = state["ff_action_384"] if row["action"] == "FF" else state["ss_action_384"]
    require(
        first["action_relative"] == action and first["forced"], "forced_action_mismatch"
    )
    for item in trajectory:
        require(
            not game.over() and item["actor"] == game.current_player,
            "trajectory_actor_mismatch",
        )
        relative = item["action_relative"]
        if len(trajectory) > 1 and item is not first:
            stones = sum(game.pits)
            require(
                item["active_pit_stones_before_move"] == stones,
                "trajectory_stone_count",
            )
            expected_backend = (
                "native_kvtb_root_action_probe_v1" if stones <= 16 else None
            )
            require(
                item["exact_root_backend"] == expected_backend, "trajectory_backend"
            )
            require(
                item["exact_root_solver_calls"] == (1 if stones <= 16 else 0),
                "trajectory_solver_calls",
            )
            require(
                isinstance(item["seed"], int) and len(item["seed_context_hash"]) == 64,
                "trajectory_seed_telemetry",
            )
        require(relative in game.possible_moves(), "trajectory_illegal_move")
        require(
            item["action_absolute"] == game.pit_index(relative),
            "trajectory_absolute_move",
        )
        require(game.move(game.pit_index(relative)), "trajectory_move_failed")
        require(item["state"] == game.to_state(), "trajectory_state_mismatch")
    require(game.over(), "trajectory_not_terminal")
    stores = game.captured_seeds
    require(sum(stores) == 48, "outcome_stone_total")
    score = (
        1.0
        if stores[root] > stores[1 - root]
        else 0.5
        if stores[root] == stores[1 - root]
        else 0.0
    )
    require(
        row["root_player"] == root and row["stores"] == stores, "outcome_store_mismatch"
    )
    require(row["score"] == score, "outcome_score_mismatch")
    require(
        row["store_margin_root_perspective"] == stores[root] - stores[1 - root],
        "outcome_margin_mismatch",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
