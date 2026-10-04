"""Read-only verifier for the exploratory FF1536 publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, seed398_ff1536_analysis as analysis
from ml.alphazero_lite import seed398_ff1536_confirmation as runner
from ml.alphazero_lite import seed398_paired_first_action as paired
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-ff1536-confirmation"


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
    reg = json.loads(before["registration"])
    require(
        reg["schema"] == "seed398-ff1536-confirmation-registration-v1",
        "registration_schema",
    )
    require(
        reg["source_sha256"] == runner.source_identities(), "execution_source_binding"
    )
    require(
        runner.verify_runtime_inputs() == reg["runtime_identities"],
        "runtime_identity_binding",
    )
    require(
        runner.sha(runner.STATES_REG) == reg["state_registration_sha256"],
        "state_registration_binding",
    )
    require(runner.sha(runner.PROBES) == reg["probe_sha256"], "frozen_probe_binding")
    states = {s["opening_index"]: s for s in runner._json(runner.STATES_REG)["states"]}
    expected_actions = runner.frozen_actions()
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
        paired.validate_outcome(wrapped["outcome"], state)
        _verify_seeds(wrapped["outcome"], state)

    ledgers = {}
    for reference, path in runner.REFERENCES.items():
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
        binding["verifier_sha256"]
        == sha(
            Path(__file__),
        ),
        "verifier_hash_binding",
    )
    after = {name: path.read_bytes() for name, path in paths.items()}
    require(before == after, "verifier_mutated_publication")
    return {
        "status": "verified",
        "new_continuations": 8,
        "reused_outcomes": 120,
        "provenance_cases": len(matrix),
        "analysis_sha256": sha(paths["analysis"]),
    }


def _verify_seeds(row: dict[str, Any], state_row: dict[str, Any]) -> None:
    game = KalahGame.from_state(state_row["state"])
    forced = state_row["ff_action_384"]
    if forced not in game.possible_moves() or not game.move(game.pit_index(forced)):
        raise ValueError("forced_action_illegal")
    for ply, move in enumerate(row["trajectory"][1:], start=1):
        state_hash = arena.canonical_game_state_hash(game)
        context = {
            "contract_version": "azlite_eval_seed_v2",
            "base_seed": 406,
            "suite_sha256": paired.SUITE_SHA256,
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
