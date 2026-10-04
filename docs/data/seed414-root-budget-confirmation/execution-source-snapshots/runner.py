"""Fresh FF384/FF1536 first-action confirmation under frozen references."""

from __future__ import annotations

import hashlib
import argparse
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, build_opening_suite as suites
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed414-root-budget-confirmation"
TMP = ROOT / ".tmp/seed414-root-budget-confirmation"
SUITE = OUT / "suite.jsonl"
REGISTRATION = OUT / "registration.json"
PROBES = OUT / "root-probes.jsonl"
TRAJECTORIES = OUT / "continuations.jsonl"
ALIASES = OUT / "aliases.jsonl"
POPULATION = ROOT / "ml/alphazero_lite/seed461_order_population.py"
PROOF = OUT / "opening-exclusion-proof.json"
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
REFERENCES = {
    "seed455": {
        "artifact": ROOT / ".tmp/seed461-order-confirmation/opponent-artifact",
        "checkpoint": ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz",
        "registration": ROOT
        / "docs/data/seed398-paired-first-action/registration.json",
    },
    "original_O0_E4": {
        "artifact": ROOT / ".tmp/seed461-order-confirmation/artifacts/O0-E4",
        "checkpoint": ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz",
        "registration": ROOT
        / "docs/data/seed398-e4-reference-diagnostic/registration.json",
    },
}
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def prepare_suite(excluded: set[str]) -> list[dict[str, Any]]:
    """Select by the established holdout contract, without model-dependent inputs."""
    population, _, _ = suites.deduplicate_openings(suites.enumerate_legal_prefixes(8))
    eligible = [
        row
        for row in suites.stratify_openings(population)
        if suites.canonical_key(row["state"]) not in excluded
        and int(row["pit_sum"]) > 32
        and not KalahGame.from_state(row["state"]).over()
    ]
    selected = suites.select_diverse(eligible, 128, 414)
    exported = [suites.export_arena_entry(row) for row in selected]
    for index, row in enumerate(exported):
        row["opening_index"] = index
        row["legal_actions"] = KalahGame.from_state(row["state"]).possible_moves()
    identities = suites.validate_arena_entries(exported)
    if len(set(identities)) != 128 or set(identities) & excluded:
        raise ValueError("fresh_suite_identity_or_exclusion_failure")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in exported)
    if SUITE.exists() and SUITE.read_text() != payload and REGISTRATION.exists():
        raise ValueError("suite_is_immutable")
    SUITE.parent.mkdir(parents=True, exist_ok=True)
    SUITE.write_text(payload)
    return exported


def frozen_runtime_identities() -> dict[str, Any]:
    result: dict[str, Any] = {"references": {}}
    for name, item in REFERENCES.items():
        reg = read_json(item["registration"])["reference"]
        hashes = {
            filename: sha(item["artifact"] / filename)
            for filename in reg["artifact_file_sha256"]
        }
        if hashes != reg["artifact_file_sha256"]:
            raise ValueError(f"artifact_hash_mismatch:{name}")
        checkpoint_hash = sha(item["checkpoint"])
        if checkpoint_hash != reg["checkpoint_sha256"]:
            raise ValueError(f"checkpoint_hash_mismatch:{name}")
        policy = read_json(item["artifact"] / "search_policy.json")
        if (
            policy.get("exact_root_threshold") != 16
            or policy.get("exact_root_objective") != "final_score_margin"
            or policy.get("exact_root_tie_rule")
            != "highest_legal_network_prior_then_lowest_move_index"
            or policy.get("exact_leaf_solve") != "disabled"
        ):
            raise ValueError(f"root16_policy_missing:{name}")
        if (
            policy.get("solver_implementation_identity")
            != "native_kvtb_root_action_probe_v1"
            or policy.get("native_probe", {}).get("sha256") != sha(NATIVE)
            or policy.get("tablebase", {}).get("sha256") != sha(TABLEBASE)
        ):
            raise ValueError(f"native_policy_runtime_identity_mismatch:{name}")
        result["references"][name] = {
            "artifact_files": hashes,
            "checkpoint_sha256": checkpoint_hash,
            "runtime_search_policy_sha256": sha(
                item["artifact"] / "search_policy.json"
            ),
        }
    expected = read_json(REFERENCES["seed455"]["registration"])["reference"]
    result["native_probe_sha256"] = sha(NATIVE)
    result["tablebase_sha256"] = sha(TABLEBASE)
    if result["native_probe_sha256"] != expected["native_probe_sha256"]:
        raise ValueError("native_probe_hash_mismatch")
    if result["tablebase_sha256"] != expected["tablebase_sha256"]:
        raise ValueError("tablebase_hash_mismatch")
    return result


def register(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze an already selected suite before any model probe is run."""
    identities = frozen_runtime_identities()
    if not REGISTRATION.exists() and any(
        path.exists() and path.stat().st_size
        for path in (PROBES, TRAJECTORIES, ALIASES)
    ):
        raise ValueError("model_evidence_exists_before_registration")
    if len(states) != 128 or len({row["state_hash"] for row in states}) != 128:
        raise ValueError("registration_suite_count_or_identity_mismatch")
    if [row["opening_index"] for row in states] != list(range(128)):
        raise ValueError("registration_suite_order_mismatch")
    if [row["state_hash"] for row in _load_jsonl(SUITE)] != [
        row["state_hash"] for row in states
    ]:
        raise ValueError("registration_suite_content_mismatch")
    suite_hash = sha(SUITE)
    reg = {
        "schema": "seed414-root-budget-confirmation-registration-v1",
        "scope": "fresh first-action quality diagnostic; not overall strength or promotion evidence",
        "suite_sha256": suite_hash,
        "suite_count": len(states),
        "states": states,
        "exclusion_proof_sha256": sha(PROOF),
        "runtime_identities": identities,
        "source_sha256": source_hashes(),
        "search": {
            "root_simulations": 1536,
            "snapshots": [384, 1536],
            "continuation_simulations": 1536,
            "c_puct": 1.25,
            "options": OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solving": "disabled",
        },
        "seed_contract": {
            "name": "azlite_eval_seed_v2",
            "base_seed": 414,
            "suite_sha256": suite_hash,
            "root_ply": 0,
            "continuation_first_ply": 1,
            "excluded_identity_fields": ["budget", "branch", "reference"],
        },
        "analysis": {
            "primary": "state-wise mean of the two reference-specific score gains, FF1536 minus FF384, averaged across all 128 states including unchanged-action zero gains",
            "bootstrap_units": 128,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 414,
            "interval": [0.025, 0.975],
            "confirm_if": "mean >= 0.03, lower interval > 0, and both reference means >= 0",
            "adaptive_extension": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(reg, indent=2, sort_keys=True) + "\n"
    if REGISTRATION.exists() and REGISTRATION.read_text() != payload:
        raise ValueError("registration_is_immutable")
    REGISTRATION.write_text(payload)
    return reg


def source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
        "opening_suite": ROOT / "ml/alphazero_lite/build_opening_suite.py",
        "population": POPULATION,
        "analysis": ROOT / "ml/alphazero_lite/seed414_root_budget_analysis.py",
        "verifier": ROOT
        / "ml/alphazero_lite/verify_seed414_root_budget_confirmation.py",
        "exclusion": ROOT / "ml/alphazero_lite/seed414_exclusion_proof.py",
        "publisher": ROOT
        / "ml/alphazero_lite/publish_seed414_root_budget_confirmation.py",
    }
    return {key: sha(path) for key, path in paths.items()}


def probe_roots(reg: dict[str, Any]) -> list[dict[str, Any]]:
    if source_hashes() != reg["source_sha256"] or sha(SUITE) != reg["suite_sha256"]:
        raise ValueError("frozen_registration_identity_changed")
    if frozen_runtime_identities() != reg["runtime_identities"]:
        raise ValueError("frozen_runtime_identity_changed")
    probes_by_index = (
        {int(row["opening_index"]): row for row in _load_jsonl(PROBES)}
        if PROBES.is_file()
        else {}
    )
    if len(probes_by_index) != len(_load_jsonl(PROBES)):
        raise ValueError("duplicate_resumed_root_probe")
    evaluator = arena.ArtifactEvaluator(REFERENCES["original_O0_E4"]["artifact"])
    for row in reg["states"]:
        if row["opening_index"] in probes_by_index:
            existing = probes_by_index[row["opening_index"]]
            if existing["state_hash"] != row["state_hash"] or existing.get(
                "registration_sha256"
            ) != sha(REGISTRATION):
                raise ValueError("resumed_root_probe_state_mismatch")
            continue
        game = KalahGame.from_state(row["state"])
        seed, context_hash = derive_search_seed(
            contract_version="azlite_eval_seed_v2",
            base_seed=414,
            suite_sha256=reg["suite_sha256"],
            opening_index=row["opening_index"],
            opening_state_hash=row["state_hash"],
            challenger_player=game.current_player,
            game_within_opening=0,
            ply=0,
            canonical_current_state_hash=arena.canonical_game_state_hash(game),
            acting_role="challenger",
        )
        started = time.perf_counter()
        result = arena.evaluate_artifact_position(
            evaluator=evaluator,
            state=row["state"],
            simulations=1536,
            seed=seed,
            c_puct=1.25,
            search_options=OPTIONS,
            exact_root_solve_threshold=16,
            root_snapshot_checkpoints={384, 1536},
        )
        elapsed = time.perf_counter() - started
        snapshots = {
            str(snap["simulation"]): {
                "action": int(snap["selected_move"]),
                "visits": {
                    str(move): int(snap["visits"][move])
                    for move in row["legal_actions"]
                },
                "moves": snap["moves"],
            }
            for snap in result.get("root_snapshots", [])
            if int(snap["simulation"]) in (384, 1536)
        }
        if set(snapshots) != {"384", "1536"}:
            raise ValueError("root_prefix_snapshots_missing")
        _validate_root_snapshots(row, snapshots)
        probe = {
            "opening_index": row["opening_index"],
            "state_hash": row["state_hash"],
            "registration_sha256": sha(REGISTRATION),
            "seed": seed,
            "seed_context_hash": context_hash,
            "snapshots": snapshots,
            "simulations": 1536,
            "c_puct": 1.25,
            "search_options": OPTIONS,
            "elapsed_wall_seconds": elapsed,
        }
        _append_jsonl(PROBES, probe)
        probes_by_index[row["opening_index"]] = probe
    if set(probes_by_index) != {row["opening_index"] for row in reg["states"]}:
        raise ValueError("root_probe_coverage_incomplete")
    return [probes_by_index[row["opening_index"]] for row in reg["states"]]


def _write_jsonl_immutable_or_empty(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"immutable_ledger_conflict:{path.name}")
    path.write_text(payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def _validate_root_snapshots(
    state: dict[str, Any], snapshots: dict[str, dict[str, Any]]
) -> None:
    legal = [int(move) for move in state["legal_actions"]]
    for budget in (384, 1536):
        key = str(budget)
        visits = snapshots[key]["visits"]
        if set(visits) != {str(move) for move in legal}:
            raise ValueError(f"root_snapshot_legal_coverage:{budget}")
        if sum(visits.values()) != budget:
            raise ValueError(f"root_snapshot_visit_total:{budget}")
        entries = {int(item["move"]): item for item in snapshots[key]["moves"]}
        winner = max(
            legal,
            key=lambda move: (
                visits[str(move)],
                float(entries[move]["q_value"]),
                float(entries[move]["prior"]),
                -move,
            ),
        )
        if winner != snapshots[key]["action"]:
            raise ValueError(f"root_snapshot_tie_rule:{budget}")
    if any(
        snapshots["384"]["visits"][move] > snapshots["1536"]["visits"][move]
        for move in snapshots["384"]["visits"]
    ):
        raise ValueError("root_snapshot_prefix_monotonicity")


def run_continuations(
    reg: dict[str, Any], probes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run each unique action/reference continuation once and record aliases."""
    if source_hashes() != reg["source_sha256"]:
        raise ValueError("frozen_execution_source_changed")
    if frozen_runtime_identities() != reg["runtime_identities"]:
        raise ValueError("frozen_runtime_identity_changed")
    by_index = {row["opening_index"]: row for row in probes}
    existing_rows = _load_jsonl(TRAJECTORIES)
    trajectories_by_key = {
        (int(row["opening_index"]), row["reference"], int(row["forced_action"])): row
        for row in existing_rows
    }
    if len(trajectories_by_key) != len(existing_rows):
        raise ValueError("duplicate_resumed_physical_continuation")
    aliases: list[dict[str, Any]] = []
    native = NativeExactRootTablebase(NATIVE, TABLEBASE, warm_on_start=True)
    try:
        for state in reg["states"]:
            index = state["opening_index"]
            probe = by_index[index]
            actions = {
                budget: int(probe["snapshots"][str(budget)]["action"])
                for budget in (384, 1536)
            }
            for reference, ref in REFERENCES.items():
                action_outcomes: dict[int, dict[str, Any]] = {}
                for budget in (384, 1536):
                    action = actions[budget]
                    if (
                        action not in action_outcomes
                        and (
                            index,
                            reference,
                            action,
                        )
                        in trajectories_by_key
                    ):
                        existing = trajectories_by_key[(index, reference, action)]
                        if existing["outcome"]["state_hash"] != state[
                            "state_hash"
                        ] or existing.get("registration_sha256") != sha(REGISTRATION):
                            raise ValueError("resumed_continuation_state_mismatch")
                        action_outcomes[action] = existing["outcome"]
                    if action not in action_outcomes:
                        outcome = play_continuation(
                            state,
                            action,
                            reference,
                            reg,
                            arena.ArtifactEvaluator(ref["artifact"]),
                            native,
                        )
                        action_outcomes[action] = outcome
                        physical = {
                            "opening_index": index,
                            "reference": reference,
                            "forced_action": action,
                            "registration_sha256": sha(REGISTRATION),
                            "outcome": outcome,
                        }
                        _append_jsonl(TRAJECTORIES, physical)
                        trajectories_by_key[(index, reference, action)] = physical
                    outcome = action_outcomes[action]
                    aliases.append(
                        {
                            "opening_index": index,
                            "state_hash": state["state_hash"],
                            "registration_sha256": sha(REGISTRATION),
                            "reference": reference,
                            "root_budget": budget,
                            "action": action,
                            "trajectory_identity": hashlib.sha256(
                                json.dumps(outcome, sort_keys=True).encode()
                            ).hexdigest(),
                            "alias_of": None
                            if budget == 384 or actions[384] != actions[1536]
                            else {
                                "opening_index": index,
                                "reference": reference,
                                "root_budget": 384,
                            },
                        }
                    )
    finally:
        native.close()
    expected = {
        (state["opening_index"], ref, action)
        for state in reg["states"]
        for ref in REFERENCES
        for action in {
            int(by_index[state["opening_index"]]["snapshots"]["384"]["action"]),
            int(by_index[state["opening_index"]]["snapshots"]["1536"]["action"]),
        }
    }
    if set(trajectories_by_key) != expected:
        raise ValueError("physical_continuation_coverage_incomplete")
    _write_jsonl_immutable_or_empty(ALIASES, aliases)
    trajectories = list(trajectories_by_key.values())
    return trajectories, aliases


def play_continuation(
    state: dict[str, Any],
    action: int,
    reference: str,
    reg: dict[str, Any],
    evaluator: arena.ArtifactEvaluator,
    native: NativeExactRootTablebase,
) -> dict[str, Any]:
    game = KalahGame.from_state(state["state"])
    root = game.current_player
    forced_absolute = game.pit_index(action)
    if action not in game.possible_moves() or not game.move(game.pit_index(action)):
        raise ValueError("forced_root_action_illegal")
    trail = [
        {
            "actor": root,
            "action_relative": action,
            "action_absolute": forced_absolute,
            "forced": True,
            "state": game.to_state(),
        }
    ]
    ply = 1
    started = time.perf_counter()
    while not game.over():
        actor = game.current_player
        seed, context_hash = derive_search_seed(
            contract_version="azlite_eval_seed_v2",
            base_seed=414,
            suite_sha256=reg["suite_sha256"],
            opening_index=state["opening_index"],
            opening_state_hash=state["state_hash"],
            challenger_player=root,
            game_within_opening=0,
            ply=ply,
            canonical_current_state_hash=arena.canonical_game_state_hash(game),
            acting_role="challenger" if actor == root else "current",
        )
        before = sum(game.pits)
        result = arena.evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=1536,
            seed=seed,
            c_puct=1.25,
            search_options=OPTIONS,
            exact_root_solve_threshold=16,
            endgame_tablebase=native,
        )
        exact = result.get("exact_root_decision")
        if before <= 16 and (exact is None or exact.get("solver_calls") != 1):
            raise ValueError("native_root16_handoff_missing")
        move = int(result["selected_move"])
        absolute = game.pit_index(move)
        if move not in game.possible_moves() or not game.move(absolute):
            raise ValueError("continuation_action_illegal")
        trail.append(
            {
                "actor": actor,
                "action_relative": move,
                "action_absolute": absolute,
                "forced": False,
                "seed": seed,
                "seed_context_hash": context_hash,
                "active_pit_stones_before_move": before,
                "exact_root_backend": "native_kvtb_root_action_probe_v1"
                if before <= 16
                else None,
                "exact_root_solver_calls": exact["solver_calls"] if exact else 0,
                "simulations": 1536,
                "c_puct": 1.25,
                "search_options": OPTIONS,
                "state": game.to_state(),
            }
        )
        ply += 1
        if ply > 1000:
            raise ValueError("continuation_safety_limit")
    margin = game.captured_seeds[root] - game.captured_seeds[1 - root]
    return {
        "opening_index": state["opening_index"],
        "state_hash": state["state_hash"],
        "root_player": root,
        "score": 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5,
        "store_margin_root_perspective": margin,
        "stores": list(game.captured_seeds),
        "trajectory": trail,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "proof",
            "select",
            "register",
            "probe",
            "continue",
            "publish",
            "verify",
        ),
    )
    args = parser.parse_args()
    if args.stage == "proof":
        from ml.alphazero_lite.seed414_exclusion_proof import publish

        proof = publish()
        print(
            json.dumps(
                {k: v for k, v in proof.items() if "identities" not in k}, indent=2
            )
        )
    elif args.stage == "select":
        proof = read_json(PROOF)
        selected = prepare_suite(set(proof["excluded_state_identities"]))
        print(json.dumps({"states": len(selected), "suite_sha256": sha(SUITE)}))
    elif args.stage == "register":
        selected = _load_jsonl(SUITE)
        registration = register(selected)
        print(json.dumps({"registered": len(selected), "sha256": sha(REGISTRATION)}))
    elif args.stage == "probe":
        registration = read_json(REGISTRATION)
        probes = probe_roots(registration)
        print(json.dumps({"root_probes": len(probes)}))
    elif args.stage == "continue":
        registration = read_json(REGISTRATION)
        probes = _load_jsonl(PROBES)
        trajectories, aliases = run_continuations(registration, probes)
        print(
            json.dumps(
                {
                    "physical_continuations": len(trajectories),
                    "logical_cases": len(aliases),
                }
            )
        )
    elif args.stage == "publish":
        from ml.alphazero_lite.publish_seed414_root_budget_confirmation import publish

        print(json.dumps(publish(), indent=2))
    else:
        from ml.alphazero_lite.verify_seed414_root_budget_confirmation import verify

        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
